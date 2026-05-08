"""
Adapter utilities for the image generation server.
"""

import base64
import io
import os

import torch
from PIL import Image as PILImage


# ── MPS memory pressure control ───────────────────────────────────────────────
# Must be set before any MPS allocations.  Prevents Metal from reserving a
# fixed high-watermark that causes paging when other processes need memory.
if torch.backends.mps.is_available():
    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


# ── Device / dtype ────────────────────────────────────────────────────────────
# Temporarily force CPU to debug memory issues
# DEVICE, DTYPE = "cpu", torch.float32
# print(f"[hf-image-server] Forcing CPU device for debugging", flush=True)

# Original device selection (uncommented to use MPS for better performance):
if torch.backends.mps.is_available():
    DEVICE, DTYPE = "mps", torch.float16
elif torch.cuda.is_available():
    DEVICE, DTYPE = "cuda", torch.float16
else:
    DEVICE, DTYPE = "cpu", torch.float32

# Allow opting into bfloat16 on MPS. Most LLM-based adapters (Qwen, GLM) use
# bfloat16 directly and are not affected by this setting.
if DEVICE == "mps" and os.environ.get("MPS_BFLOAT16") == "1":
    DTYPE = torch.bfloat16

# torch.compile on CUDA gives 20-40% throughput after the first inference.
# Disable with COMPILE_MODEL=0 if you see trace errors.
_COMPILE_ENABLED = DEVICE == "cuda" and os.environ.get("COMPILE_MODEL", "1") != "0"

print(f"[hf-image-server] device={DEVICE} dtype={DTYPE} compile={_COMPILE_ENABLED}", flush=True)


# ── Image encode/decode helpers ───────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _decode_image(data_url: str) -> PILImage.Image:
    """Decode a base64 data URL, raw base64 string, local file path, or HTTP URL to an RGB PIL Image."""
    s = data_url.strip()

    # Local file path: /output/filename.jpg  (web path served by the Node bridge)
    if s.startswith("/"):
        file_path = os.path.join(_SCRIPT_DIR, s.lstrip("/"))
        return PILImage.open(file_path).convert("RGB")

    # Absolute HTTP/HTTPS URL
    if s.startswith("http://") or s.startswith("https://"):
        import urllib.request
        with urllib.request.urlopen(s) as resp:  # noqa: S310
            return PILImage.open(io.BytesIO(resp.read())).convert("RGB")

    # Base64 data URL or raw base64
    _, b64 = s.split(",", 1) if "," in s else ("", s)
    # Normalise to standard base64:
    # 1. Remove ALL whitespace (including embedded newlines from multi-line
    #    encoders) before length calculation — strip() only removes edges.
    # 2. Convert URL-safe chars (-→+ and _→/).
    # 3. Strip any existing padding, then re-add the correct amount so that
    #    len(b64) always equals 0 mod 4 without double-padding.
    b64 = "".join(b64.split()).replace("-", "+").replace("_", "/")
    b64 = b64.rstrip("=")
    b64 += "=" * (-len(b64) % 4)
    return PILImage.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


def _encode_image(img: PILImage.Image) -> str:
    """
    Encode a PIL Image to a base64 JPEG data URL (quality 90).
    JPEG is 5-10× smaller than PNG for photographic content; quality 90 is
    visually lossless and the format difference is transparent to the browser.
    """
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=90, optimize=True)
    return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"


# ── Step-level progress tracking ──────────────────────────────────────────────
# Written by _step_callback (inference thread), read by GET /progress (any thread).
# CPython's GIL makes plain dict writes/reads atomic enough for this use-case.
_progress: dict = {"step": 0, "total": 0}


def _step_callback(pipe, step: int, timestep, callback_kwargs: dict) -> dict:
    """callback_on_step_end handler — updates the global progress counter."""
    _progress["step"] = step + 1
    return callback_kwargs


# ── Pipeline optimisation helpers ─────────────────────────────────────────────
def _apply_pipeline_optimizations(pipe) -> None:
    """Apply inference-time speed and memory optimisations to a loaded pipeline."""

    # Suppress tqdm — eliminates I/O overhead on every denoising step
    if hasattr(pipe, "set_progress_bar_config"):
        pipe.set_progress_bar_config(disable=True)

    # Attention slicing: splits attention heads into chunks, reducing peak VRAM.
    # On MPS this also reduces Metal command-buffer pressure.
    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing(1)  # 1 = maximum (most aggressive) slicing

    # VAE slicing: decodes the latent image tile-by-tile — prevents OOM at
    # larger resolutions (768×768+) without quality loss.
    if hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()

    if DEVICE == "cuda":
        # xformers provides a fused, memory-efficient attention kernel on CUDA
        if hasattr(pipe, "enable_xformers_memory_efficient_attention"):
            try:
                pipe.enable_xformers_memory_efficient_attention()
                print("[hf-image-server] xformers attention enabled", flush=True)
            except Exception as e:
                print(f"[hf-image-server] xformers unavailable: {e}", flush=True)

        # torch.compile: traces the UNet/transformer once and emits an optimised
        # CUDA kernel — pays a one-time compilation cost on the first inference.
        if _COMPILE_ENABLED:
            for attr in ("unet", "transformer"):
                _try_compile_attr(pipe, attr)

    elif DEVICE == "mps":
        # Flush pending Metal ops so they don't interfere with the first inference
        torch.mps.synchronize()


def _call_with_progress(pipe, **kwargs):
    """Call a pipeline, injecting the step callback when supported."""
    print("[hf-image-server] inference:", {k: v for k, v in kwargs.items() if k not in ("prompt", "image", "generator")}, flush=True)
    try:
        return pipe(**kwargs, callback_on_step_end=_step_callback)
    except TypeError:
        return pipe(**kwargs)


def _try_compile_attr(pipe, attr: str) -> None:
    module = getattr(pipe, attr, None)
    if module is None:
        return
    try:
        setattr(pipe, attr, torch.compile(module, mode="reduce-overhead"))
        print(f"[hf-image-server] torch.compile applied to {attr}", flush=True)
    except Exception as e:
        print(f"[hf-image-server] torch.compile({attr}) skipped: {e}", flush=True)


# ── Inference step sentinel helpers ───────────────────────────────────────────
def _effective_steps(requested: int, default: int) -> int:
    """Substitute adapter default when the caller used the API sentinel value (4)."""
    return default if requested == 4 else requested


def _effective_guidance(requested: float, default: float) -> float:
    """Substitute adapter default when the caller used the API sentinel value (0.0)."""
    return default if requested == 0.0 else requested