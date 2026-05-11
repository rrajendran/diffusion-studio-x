"""
Shared utilities: device selection, image encode/decode, progress tracking.
Mirrors hf-image-server/adaptors/utils.py but adapted for the lms-image-server.
"""

import base64
import io
import os

import torch
from PIL import Image as PILImage


# ── MPS memory pressure control ───────────────────────────────────────────────
if torch.backends.mps.is_available():
    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


# ── Device / dtype ────────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    DEVICE, DTYPE = "mps", torch.float16
elif torch.cuda.is_available():
    DEVICE, DTYPE = "cuda", torch.float16
else:
    DEVICE, DTYPE = "cpu", torch.float32

print(f"[lms-image-server] device={DEVICE} dtype={DTYPE}", flush=True)


# ── Image encode / decode ─────────────────────────────────────────────────────
def _decode_image(data_url: str) -> PILImage.Image:
    s = data_url.strip()
    if s.startswith("http://") or s.startswith("https://"):
        import urllib.request
        with urllib.request.urlopen(s) as resp:  # noqa: S310
            return PILImage.open(io.BytesIO(resp.read())).convert("RGB")
    _, b64 = s.split(",", 1) if "," in s else ("", s)
    b64 = "".join(b64.split()).replace("-", "+").replace("_", "/")
    b64 = b64.rstrip("=")
    b64 += "=" * (-len(b64) % 4)
    return PILImage.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


def _encode_image(img: PILImage.Image) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=90, optimize=True)
    return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"


# ── Step-level progress tracking ──────────────────────────────────────────────
_progress: dict = {"step": 0, "total": 0}


def _step_callback(pipe, step: int, timestep, callback_kwargs: dict) -> dict:
    _progress["step"] = step + 1
    return callback_kwargs


# ── Inference sentinel helpers ─────────────────────────────────────────────────
def _effective_steps(requested: int, default: int) -> int:
    return default if requested == 4 else requested


def _effective_guidance(requested: float, default: float) -> float:
    return default if requested == 0.0 else requested


# ── Pipeline optimisation helpers ─────────────────────────────────────────────
def _apply_pipeline_optimizations(pipe) -> None:
    if hasattr(pipe, "set_progress_bar_config"):
        pipe.set_progress_bar_config(disable=True)
    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing(1)
    if hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()
    if DEVICE == "mps":
        torch.mps.synchronize()


def _call_with_progress(pipe, **kwargs):
    try:
        return pipe(**kwargs, callback_on_step_end=_step_callback)
    except TypeError:
        return pipe(**kwargs)
