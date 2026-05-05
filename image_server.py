"""
Local image generation server — port 8001.

Architecture: a ModelRegistry maps model IDs to ModelAdapter instances.
Each adapter encapsulates loading, default hyperparameters, and inference
for one model family, keeping the REST layer free of per-model branching.

Supported adapters (first match wins):
  GLMImageAdapter   — zai-org/GLM-Image  (text-to-image + image-to-image)
  QwenImageAdapter  — Qwen/Qwen-Image-*  (text-to-image + image-to-image)
  DiffusionAdapter  — everything else    (text-to-image only, generic fallback)

Speed optimisations applied at load time:
  - Attention slicing (all devices): reduces peak memory → fewer MPS/GPU stalls
  - VAE slicing (all devices): prevents OOM on larger resolutions
  - Progress bar disabled: eliminates tqdm I/O overhead per step
  - xformers memory-efficient attention (CUDA): faster attention kernel
  - torch.compile UNet/transformer (CUDA, opt-out via COMPILE_MODEL=0)
  - MPS high-watermark env override: reduces Metal memory pressure
  - Inference lock: serialises requests so they don't fight over GPU memory
  - Warmup pass (MPS/CUDA): primes Metal shader or CUDA JIT before first request
  - JPEG output (quality 90): 5-10× smaller payload vs PNG, fully transparent to clients
"""

import abc
import base64
import gc
import io
import os
import threading
import time
import traceback

import random

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image as PILImage
from pydantic import BaseModel, Field


# ── MPS memory pressure control ───────────────────────────────────────────────
# Must be set before any MPS allocations.  Prevents Metal from reserving a
# fixed high-watermark that causes paging when other processes need memory.
if torch.backends.mps.is_available():
    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


# ── Device / dtype ────────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    DEVICE, DTYPE = "mps", torch.float32
elif torch.cuda.is_available():
    DEVICE, DTYPE = "cuda", torch.float16
else:
    DEVICE, DTYPE = "cpu", torch.float32

# Allow opting into float16 on MPS (faster on M2+, may produce black images on
# older silicon or some models).  Set MPS_FLOAT16=1 to enable.
if DEVICE == "mps" and os.environ.get("MPS_FLOAT16") == "1":
    DTYPE = torch.float16

# torch.compile on CUDA gives 20-40% throughput after the first inference.
# Disable with COMPILE_MODEL=0 if you see trace errors.
_COMPILE_ENABLED = DEVICE == "cuda" and os.environ.get("COMPILE_MODEL", "1") != "0"

print(f"[image-server] device={DEVICE} dtype={DTYPE} compile={_COMPILE_ENABLED}", flush=True)


# ── Image encode/decode helpers ───────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

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
                print("[image-server] xformers attention enabled", flush=True)
            except Exception as e:
                print(f"[image-server] xformers unavailable: {e}", flush=True)

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
    print("[image-server] inference:", {k: v for k, v in kwargs.items() if k not in ("prompt", "image", "generator")}, flush=True)
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
        print(f"[image-server] torch.compile applied to {attr}", flush=True)
    except Exception as e:
        print(f"[image-server] torch.compile({attr}) skipped: {e}", flush=True)


# ── Inference step sentinel helpers ───────────────────────────────────────────
def _effective_steps(requested: int, default: int) -> int:
    """Substitute adapter default when the caller used the API sentinel value (4)."""
    return default if requested == 4 else requested


def _effective_guidance(requested: float, default: float) -> float:
    """Substitute adapter default when the caller used the API sentinel value (0.0)."""
    return default if requested == 0.0 else requested


# ── ModelAdapter base class ───────────────────────────────────────────────────
class ModelAdapter(abc.ABC):
    """
    Contract for a model family adapter.

    Subclasses must implement:
      matches()        — decides ownership of a model_id
      load()           — returns (t2i_pipe, i2i_pipe); i2i_pipe may be None
      text_to_image()  — runs inference and returns a PIL Image

    Subclasses may implement:
      image_to_image() — required when supports_i2i = True

    Class-level attributes set sensible per-family defaults for steps/guidance.
    """

    default_steps: int = 4
    default_guidance: float = 0.0
    supports_i2i: bool = False

    @classmethod
    @abc.abstractmethod
    def matches(cls, model_id: str) -> bool:
        """Return True if this adapter handles the given model_id."""

    @abc.abstractmethod
    def load(self, model_id: str) -> tuple:
        """
        Load and return (t2i_pipe, i2i_pipe).
        i2i_pipe may be None for text-only models.
        Call _apply_pipeline_optimizations() before returning.
        """

    @abc.abstractmethod
    def text_to_image(self, pipes: tuple, req) -> PILImage.Image:
        """Run text-to-image inference; return a PIL Image."""

    def image_to_image(self, pipes: tuple, req, ref_img: PILImage.Image) -> PILImage.Image:
        raise ValueError(f"Model family does not support image-to-image: {req.model}")

    def capabilities(self) -> dict:
        return {
            "supports_i2i": self.supports_i2i,
            "default_steps": self.default_steps,
            "default_guidance": self.default_guidance,
        }


# ── GLM-Image adapter ─────────────────────────────────────────────────────────
class GLMImageAdapter(ModelAdapter):
    """Adapter for zai-org/GLM-Image (flow-matching model)."""

    default_steps = 50
    default_guidance = 5.0
    supports_i2i = True

    @classmethod
    def matches(cls, model_id: str) -> bool:
        return "GLM-Image" in model_id

    def load(self, model_id: str) -> tuple:
        from diffusers.pipelines.glm_image import GlmImagePipeline

        load_kwargs = {"torch_dtype": torch.bfloat16}
        if DEVICE == "cuda":
            load_kwargs["device_map"] = "auto"

        pipe = GlmImagePipeline.from_pretrained(model_id, **load_kwargs)

        if DEVICE == "cuda":
            pass  # device_map="auto" already handled placement
        elif DEVICE == "mps":
            # GLM-Image has two large sub-models (AR transformer + DiT).
            # Loading both onto the Metal heap at once exhausts unified memory
            # and triggers an OOM SIGKILL.  enable_model_cpu_offload() keeps
            # weights on CPU and moves each component to MPS only for its
            # forward pass, reducing peak Metal allocation by ~50%.
            pipe.enable_model_cpu_offload()
        else:
            pipe = pipe.to(DEVICE)

        _apply_pipeline_optimizations(pipe)
        return pipe, pipe  # same pipeline handles both t2i and i2i modes

    def text_to_image(self, pipes: tuple, req) -> PILImage.Image:
        pipe, _ = pipes
        return _call_with_progress(
            pipe,
            prompt=req.prompt,
            height=req.height,
            width=req.width,
            num_inference_steps=_effective_steps(req.num_inference_steps, self.default_steps),
            guidance_scale=_effective_guidance(req.guidance_scale, self.default_guidance),
        ).images[0]

    def image_to_image(self, pipes: tuple, req, ref_img: PILImage.Image) -> PILImage.Image:
        pipe, _ = pipes
        # GLM-Image encodes the reference image with position embeddings that must
        # match the target output size — mismatched sizes cause a tensor shape error.
        if ref_img.size != (req.width, req.height):
            ref_img = ref_img.resize((req.width, req.height), PILImage.LANCZOS)

        if DEVICE == "mps":
            # enable_model_cpu_offload() keeps weights on CPU but the image
            # processor places pixel_values on MPS, causing a device mismatch
            # inside vision_language_encoder.  Temporarily move the encoder to
            # MPS for the i2i call so weights and inputs are on the same device.
            pipe.vision_language_encoder.to("mps")
            try:
                result = _call_with_progress(
                    pipe,
                    prompt=req.prompt,
                    image=[ref_img],
                    height=req.height,
                    width=req.width,
                    num_inference_steps=_effective_steps(req.num_inference_steps, self.default_steps),
                    guidance_scale=_effective_guidance(req.guidance_scale, self.default_guidance),
                ).images[0]
            finally:
                pipe.vision_language_encoder.to("cpu")
                torch.mps.empty_cache()
            return result

        return _call_with_progress(
            pipe,
            prompt=req.prompt,
            image=[ref_img],
            height=req.height,
            width=req.width,
            num_inference_steps=_effective_steps(req.num_inference_steps, self.default_steps),
            guidance_scale=_effective_guidance(req.guidance_scale, self.default_guidance),
        ).images[0]


# ── Qwen-Image adapter ────────────────────────────────────────────────────────
class QwenImageAdapter(ModelAdapter):
    """
    Adapter for Qwen/Qwen-Image-* models.
    t2i: AutoPipelineForText2Image (e.g. Qwen/Qwen-Image-2512)
    i2i: QwenImageEditPlusPipeline (e.g. Qwen/Qwen-Image-Edit-2511)
    """

    default_steps = 20
    default_guidance = 4.0
    supports_i2i = True

    @classmethod
    def matches(cls, model_id: str) -> bool:
        return "Qwen" in model_id and "Image" in model_id

    def load(self, model_id: str) -> tuple:
        from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image, QwenImageEditPlusPipeline

        dtype = torch.bfloat16 if DEVICE != "cpu" else torch.float32
        pipe_t2i = AutoPipelineForText2Image.from_pretrained(
            model_id,
            torch_dtype=dtype,
            trust_remote_code=True,
        ).to(DEVICE)

        _apply_pipeline_optimizations(pipe_t2i)

        pipe_i2i = None
        try:
            pipe_i2i = QwenImageEditPlusPipeline.from_pipe(pipe_t2i)
            print(f"[image-server] i2i: QwenImageEditPlusPipeline", flush=True)
        except Exception as e:
            print(f"[image-server] QwenImageEditPlusPipeline unavailable ({e}), trying AutoPipelineForImage2Image", flush=True)
            try:
                pipe_i2i = AutoPipelineForImage2Image.from_pipe(pipe_t2i)
                print(f"[image-server] i2i: AutoPipelineForImage2Image", flush=True)
            except Exception as e2:
                print(f"[image-server] i2i pipeline unavailable for {model_id}: {e2}", flush=True)

        if pipe_i2i is not None:
            _apply_pipeline_optimizations(pipe_i2i)

        return pipe_t2i, pipe_i2i

    def text_to_image(self, pipes: tuple, req) -> PILImage.Image:
        pipe_t2i, _ = pipes
        # Qwen-Image uses true_cfg_scale for CFG (not guidance_scale which it ignores).
        # negative_prompt="" is required to activate CFG; omitting it silences true_cfg_scale too.
        return _call_with_progress(
            pipe_t2i,
            prompt=req.prompt,
            negative_prompt="",
            height=req.height,
            width=req.width,
            num_inference_steps=_effective_steps(req.num_inference_steps, self.default_steps),
            true_cfg_scale=_effective_guidance(req.guidance_scale, self.default_guidance),
        ).images[0]

    def image_to_image(self, pipes: tuple, req, ref_img: PILImage.Image) -> PILImage.Image:
        _, pipe_i2i = pipes
        if pipe_i2i is None:
            raise ValueError("Qwen i2i pipeline unavailable")
        return _call_with_progress(
            pipe_i2i,
            image=[ref_img],
            prompt=req.prompt,
            generator=torch.manual_seed(req.seed),
            true_cfg_scale=_effective_guidance(req.guidance_scale, self.default_guidance),
            negative_prompt=" ",
            guidance_scale=1.0,
            num_inference_steps=_effective_steps(req.num_inference_steps, self.default_steps),
            num_images_per_prompt=1,
            height=req.height,
            width=req.width,
        ).images[0]


# ── Generic diffusers adapter (fallback) ──────────────────────────────────────
class DiffusionAdapter(ModelAdapter):
    """
    Fallback for any model loadable via DiffusionPipeline (e.g. turbo models).
    Supports text-to-image only; passes through num_inference_steps as-is so
    turbo models (which need exactly 4 steps) work correctly.
    """

    default_steps = 4
    default_guidance = 0.0
    supports_i2i = False

    @classmethod
    def matches(cls, model_id: str) -> bool:
        return True  # catch-all — must be last in registry

    def load(self, model_id: str) -> tuple:
        from diffusers import DiffusionPipeline

        pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=DTYPE).to(DEVICE)
        _apply_pipeline_optimizations(pipe)
        return pipe, None

    def text_to_image(self, pipes: tuple, req) -> PILImage.Image:
        pipe, _ = pipes
        return _call_with_progress(
            pipe,
            prompt=req.prompt,
            width=req.width,
            height=req.height,
            num_inference_steps=req.num_inference_steps,
            guidance_scale=req.guidance_scale,
        ).images[0]


# ── ModelRegistry ─────────────────────────────────────────────────────────────
_REGISTRY: list[ModelAdapter] = [
    GLMImageAdapter(),
    QwenImageAdapter(),
    DiffusionAdapter(),  # catch-all — must be last
]


def get_adapter(model_id: str) -> ModelAdapter:
    for adapter in _REGISTRY:
        if adapter.matches(model_id):
            return adapter
    return _REGISTRY[-1]


# ── Pipeline cache (LRU-1) ────────────────────────────────────────────────────
_cached_model_id: str | None = None
_cached_pipes: tuple | None = None


def _detect_model_source(model_id: str) -> str:
    """Return a short label describing where the model will be loaded from."""
    if os.path.isdir(model_id):
        return f"local path ({model_id})"
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(model_id, local_files_only=True)
        return "local HF cache"
    except Exception:
        return "HuggingFace Hub (download)"


def _warmup_pipeline(pipes: tuple, adapter: ModelAdapter) -> None:
    """
    Run a minimal 2-step inference to prime Metal shader compilation (MPS) or
    CUDA JIT caches.  Uses 2 steps — just enough for schedulers to run —
    so warmup finishes in seconds rather than a full inference pass.
    """
    if DEVICE not in ("mps", "cuda"):
        return

    class _Req:
        prompt = "warmup"
        width = 512
        height = 512
        num_inference_steps = 2
        guidance_scale = adapter.default_guidance

    t0 = time.time()
    print("[image-server] warming up …", flush=True)
    try:
        with torch.inference_mode():
            adapter.text_to_image(pipes, _Req())
        if DEVICE == "mps":
            torch.mps.synchronize()
        print(f"[image-server] warmup done ({time.time() - t0:.1f}s)", flush=True)
    except Exception as e:
        print(f"[image-server] warmup skipped: {e}", flush=True)


def get_pipes(model_id: str) -> tuple:
    global _cached_model_id, _cached_pipes

    if _cached_pipes is not None and _cached_model_id == model_id:
        return _cached_pipes

    if _cached_pipes is not None:
        print(f"[image-server] unloading {_cached_model_id} …", flush=True)
        del _cached_pipes
        _cached_pipes = None
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        elif DEVICE == "mps":
            torch.mps.empty_cache()

    source = _detect_model_source(model_id)
    print(f"[image-server] loading {model_id} | source={source}", flush=True)
    t0 = time.time()
    adapter = get_adapter(model_id)
    _cached_pipes = adapter.load(model_id)
    _cached_model_id = model_id
    print(f"[image-server] ready ({time.time() - t0:.1f}s) | source={source}", flush=True)

    _warmup_pipeline(_cached_pipes, adapter)
    return _cached_pipes


# ── Inference serialisation lock ──────────────────────────────────────────────
# MPS and single-GPU CUDA have one hardware queue.  Concurrent requests compete
# for the same memory and serialise themselves badly.  An explicit lock ensures
# clean sequential execution with predictable latency.
_inference_lock = threading.Lock()

# ── Step-level progress tracking ──────────────────────────────────────────────
# Written by _step_callback (inference thread), read by GET /progress (any thread).
# CPython's GIL makes plain dict writes/reads atomic enough for this use-case.
_progress: dict = {"step": 0, "total": 0}


def _step_callback(pipe, step: int, timestep, callback_kwargs: dict) -> dict:
    """callback_on_step_end handler — updates the global progress counter."""
    _progress["step"] = step + 1
    return callback_kwargs


DEFAULT_MODEL = os.environ.get("IMAGE_MODEL", "")

# Optional startup pre-load — if IMAGE_MODEL is set and the model is cached
# locally, load it now to warm Metal/CUDA before the first request.
# Failures are non-fatal so the server always binds port 8001.
if DEFAULT_MODEL:
    try:
        get_pipes(DEFAULT_MODEL)
    except Exception as _e:
        print(f"[image-server] startup pre-load skipped: {_e}", flush=True)


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class GenerateRequest(BaseModel):
    prompt: str
    model: str = ""
    width: int = 512
    height: int = 512
    num_inference_steps: int = 4    # sentinel — adapters substitute their own default
    guidance_scale: float = 0.0     # sentinel — adapters substitute their own default
    seed: int = Field(default_factory=lambda: random.randint(0, 2**32 - 1))
    reference_image: str | None = None  # base64 data URL; triggers i2i when present


@app.get("/health")
def health():
    return {"status": "ok", "model": _cached_model_id, "device": DEVICE}


@app.get("/model-capabilities")
def model_capabilities():
    return [
        {"family": "GLM-Image",  "pattern": "GLM-Image",   **GLMImageAdapter().capabilities()},
        {"family": "Qwen-Image", "pattern": "Qwen.*Image",  **QwenImageAdapter().capabilities()},
        {"family": "Diffusion",  "pattern": "*",            **DiffusionAdapter().capabilities()},
    ]


@app.get("/progress")
def get_progress():
    return _progress


def _ms(t: float) -> str:
    return f"{(time.time() - t) * 1000:.0f}ms"


@app.post("/generate")
def generate(req: GenerateRequest):
    if not req.model:
        raise HTTPException(status_code=400, detail="model is required")

    # Reset progress immediately so any poll during model load sees a clean state
    # rather than the stale 100% from the previous generation.
    _progress.update({"step": 0, "total": 0})

    t_req = time.time()
    req_id = int(t_req * 1000) % 1_000_000
    has_ref = bool(req.reference_image)
    adapter = get_adapter(req.model)
    effective_steps = _effective_steps(req.num_inference_steps, adapter.default_steps)

    print(
        f"[gen:{req_id}] ▶ request | model={req.model} {req.width}x{req.height} "
        f"steps={effective_steps} ref={has_ref} adapter={type(adapter).__name__}",
        flush=True,
    )

    t_load = time.time()
    cache_hit = _cached_model_id == req.model
    try:
        pipes = get_pipes(req.model)
    except Exception as e:
        print(f"[gen:{req_id}] ✗ load failed ({_ms(t_load)}): {e}", flush=True)
        raise HTTPException(status_code=500, detail=f"Failed to load model '{req.model}': {e}")
    print(f"[gen:{req_id}]   load {'(cached)' if cache_hit else '(cold)'} {_ms(t_load)}", flush=True)

    _progress.update({"step": 0, "total": effective_steps})
    mode = "t2i"

    t_lock = time.time()
    with _inference_lock:
        print(f"[gen:{req_id}]   lock wait {_ms(t_lock)}", flush=True)
        try:
            with torch.inference_mode():
                if req.reference_image:
                    if adapter.supports_i2i:
                        mode = "i2i"
                        t_dec = time.time()
                        ref_img = _decode_image(req.reference_image)
                        print(f"[gen:{req_id}]   decode ref {_ms(t_dec)} | size={ref_img.size}", flush=True)
                        t_inf = time.time()
                        image = adapter.image_to_image(pipes, req, ref_img)
                        print(f"[gen:{req_id}]   i2i inference {_ms(t_inf)}", flush=True)
                    else:
                        print(f"[gen:{req_id}]   no i2i support → t2i fallback", flush=True)
                        t_inf = time.time()
                        image = adapter.text_to_image(pipes, req)
                        print(f"[gen:{req_id}]   t2i inference {_ms(t_inf)}", flush=True)
                else:
                    t_inf = time.time()
                    image = adapter.text_to_image(pipes, req)
                    print(f"[gen:{req_id}]   t2i inference {_ms(t_inf)}", flush=True)
        except Exception as e:
            print(f"[gen:{req_id}] ✗ inference error ({mode}) after {_ms(t_req)}: {e}", flush=True)
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Inference error ({mode}): {e}")

    t_enc = time.time()
    image_url = _encode_image(image)
    print(f"[gen:{req_id}]   encode {_ms(t_enc)} | size={image.size}", flush=True)

    elapsed = round(time.time() - t_req, 2)
    print(f"[gen:{req_id}] ✓ done | mode={mode} total={elapsed}s seed={req.seed}", flush=True)
    return {
        "imageUrl": image_url,
        "mode": mode,
        "elapsed": elapsed,
        "meta": {
            "model": req.model,
            "seed": req.seed,
            "steps": effective_steps,
            "width": req.width,
            "height": req.height,
            "guidance_scale": round(_effective_guidance(req.guidance_scale, adapter.default_guidance), 2),
        },
    }


# ── Chat completions (OpenAI-compatible) ─────────────────────────────────────
# Used by the Enhance feature when the HuggingFace provider is active.
# Loads the requested model as a causal-LM text pipeline (separate from the
# image pipeline) and streams a single non-streaming response.

_chat_model_id: str | None = None
_chat_pipeline = None
_chat_lock = threading.Lock()


def _get_chat_pipeline(model_id: str):
    global _chat_model_id, _chat_pipeline
    if _chat_model_id == model_id and _chat_pipeline is not None:
        return _chat_pipeline
    print(f"[chat] loading text model {model_id!r} ...", flush=True)
    from transformers import pipeline as hf_pipeline
    _chat_pipeline = hf_pipeline(
        "text-generation",
        model=model_id,
        device_map="auto",
        dtype=DTYPE,
    )
    _chat_model_id = model_id
    print(f"[chat] model ready", flush=True)
    return _chat_pipeline


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = ""
    messages: list[ChatMessage]
    max_tokens: int = 512
    temperature: float = 0.7
    stream: bool = False


@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest):
    if not req.model:
        raise HTTPException(status_code=400, detail="model is required")
    t0 = time.time()
    try:
        with _chat_lock:
            pipe = _get_chat_pipeline(req.model)
            msgs = [{"role": m.role, "content": m.content} for m in req.messages]
            out = pipe(msgs, max_new_tokens=req.max_tokens, temperature=req.temperature, do_sample=True)
        # transformers text-gen pipeline returns generated tokens after the prompt
        generated = out[0].get("generated_text", "")
        if isinstance(generated, list):
            # chat-template output: last message is the assistant turn
            content = generated[-1].get("content", "") if generated else ""
        else:
            content = str(generated)
        elapsed = round(time.time() - t0, 2)
        print(f"[chat] done in {elapsed}s | model={req.model}", flush=True)
        return {
            "id": f"chatcmpl-{int(t0*1000)}",
            "object": "chat.completion",
            "model": req.model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
            "usage": {},
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Chat completion error: {e}")


if __name__ == "__main__":
    port = int(os.environ.get("IMAGE_SERVER_PORT", 8001))
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
