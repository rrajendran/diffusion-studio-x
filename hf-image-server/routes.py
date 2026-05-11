"""
FastAPI routes for the image generation server.
"""

from fastapi.middleware.cors import CORSMiddleware
import concurrent.futures
import os
import time
import traceback

import torch
from fastapi import FastAPI, HTTPException

from adaptors import (
    DEVICE,
    _effective_steps,
    _effective_guidance,
    _decode_image,
    _encode_image,
    _encode_video,
    _progress,
    get_adapter,
    GLMImageAdapter,
    QwenImageAdapter,
    FluxAdapter,
    FluxKleinAdapter,
    FluxKleinKVAdapter,
    FluxKontextAdapter,
    ZImageAdapter,
    ErnieImageAdapter,
    SD3Adapter,
    WanAdapter,
    DiffusionAdapter,
)
from domain import (
    GenerateRequest,
    PreloadRequest,
    ChatCompletionRequest,
)
from services import service


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok", "model": service.get_cached_model_id(), "device": DEVICE}


@app.get("/cached-models")
def cached_models():
    """Return a list of model IDs that are fully cached locally via huggingface_hub."""
    try:
        from huggingface_hub import scan_cache_dir
        info = scan_cache_dir()
        return [r.repo_id for r in info.repos if r.repo_type == "model"]
    except Exception as e:
        print(f"[hf-image-server] cached-models error: {e}", flush=True)
        return []


@app.get("/model-capabilities")
def model_capabilities():
    return [
        {"family": "GLM-Image",  "pattern": "GLM-Image",             **GLMImageAdapter().capabilities()},
        {"family": "Qwen-Image", "pattern": "Qwen.*Image",           **QwenImageAdapter().capabilities()},
        {"family": "FLUX",       "pattern": "FLUX\\.1-(schnell|dev)", **FluxAdapter().capabilities()},
        {"family": "FLUX-Klein",   "pattern": "FLUX\\.2-klein(?!.*kv)", **FluxKleinAdapter().capabilities()},
        {"family": "FLUX-Klein-KV", "pattern": "FLUX\\.2-klein.*kv",       **FluxKleinKVAdapter().capabilities()},
        {"family": "FLUX-Kontext", "pattern": "FLUX\.1-Kontext",            **FluxKontextAdapter().capabilities()},
        {"family": "Z-Image",    "pattern": "Z-Image",                  **ZImageAdapter().capabilities()},
        {"family": "ERNIE-Image", "pattern": "ERNIE-Image",              **ErnieImageAdapter().capabilities()},
        {"family": "SD3",        "pattern": "stable-diffusion-3",        **SD3Adapter().capabilities()},
        {"family": "Wan-Video",  "pattern": "Wan-AI/Wan",               **WanAdapter().capabilities()},
        {"family": "Diffusion",  "pattern": "stable-diffusion-xl*",      **DiffusionAdapter().capabilities()},
    ]


@app.get("/progress")
def get_progress():
    return _progress


def _ms(t: float) -> str:
    return f"{(time.time() - t) * 1000:.0f}ms"


@app.post("/preload")
def preload(req: PreloadRequest):
    """Eagerly load a model into memory so the first /generate is fast.
    Returns immediately with {"status": "loading"} and warms up in the background."""
    import threading

    def _load():
        try:
            service.get_pipes(req.model)
            print(
                f"[hf-image-server] preload complete: {req.model}", flush=True)
        except Exception as e:
            print(f"[hf-image-server] preload failed: {e}", flush=True)
    threading.Thread(target=_load, daemon=True).start()
    return {"status": "loading", "model": req.model}


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
    effective_steps = _effective_steps(
        req.num_inference_steps, adapter.default_steps)

    print(
        f"[gen:{req_id}] ▶ request | model={req.model} {req.width}x{req.height} "
        f"steps={effective_steps} ref={has_ref} adapter={type(adapter).__name__}",
        flush=True,
    )

    t_load = time.time()
    cache_hit = service.get_cached_model_id() == req.model
    try:
        pipes = service.get_pipes(req.model)
    except Exception as e:
        print(f"[gen:{req_id}] ✗ load failed ({_ms(t_load)}): {e}", flush=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to load model '{req.model}': {e}")
    print(
        f"[gen:{req_id}]   load {'(cached)' if cache_hit else '(cold)'} {_ms(t_load)}", flush=True)

    _progress.update({"step": 0, "total": effective_steps})
    is_video = adapter.is_video
    mode = "t2v" if is_video else "t2i"

    # Server-side inference timeout. Video generation (Wan2.2-14B + cpu_offload on MPS)
    # runs ~2-5 min/step due to 14B weight I/O; 5 steps can take 10-25 min total.
    # Default: 60 min for video, 20 min for images. Override via VIDEO_INFERENCE_TIMEOUT
    # / INFERENCE_TIMEOUT env vars.
    if is_video:
        _INFERENCE_TIMEOUT = int(os.environ.get("VIDEO_INFERENCE_TIMEOUT", "3600"))
    else:
        _INFERENCE_TIMEOUT = int(os.environ.get("INFERENCE_TIMEOUT", "1200"))

    t_lock = time.time()
    with service.get_inference_lock():
        print(f"[gen:{req_id}]   lock wait {_ms(t_lock)}", flush=True)
        try:
            def _run_inference():
                nonlocal mode
                with torch.inference_mode():
                    if req.reference_image:
                        if adapter.supports_i2i:
                            mode = "i2v" if is_video else "i2i"
                            t_dec = time.time()
                            ref_img = _decode_image(req.reference_image)
                            print(
                                f"[gen:{req_id}]   decode ref {_ms(t_dec)} | size={ref_img.size}", flush=True)
                            t_inf = time.time()
                            if is_video:
                                result = adapter.image_to_video(pipes, req, ref_img)
                            else:
                                result = adapter.image_to_image(pipes, req, ref_img)
                            print(
                                f"[gen:{req_id}]   {mode} inference {_ms(t_inf)}", flush=True)
                        else:
                            print(
                                f"[gen:{req_id}]   no i2i support → t2i fallback", flush=True)
                            t_inf = time.time()
                            if is_video:
                                result = adapter.text_to_video(pipes, req)
                            else:
                                result = adapter.text_to_image(pipes, req)
                            print(
                                f"[gen:{req_id}]   t2i inference {_ms(t_inf)}", flush=True)
                    else:
                        t_inf = time.time()
                        if is_video:
                            result = adapter.text_to_video(pipes, req)
                        else:
                            result = adapter.text_to_image(pipes, req)
                        print(
                            f"[gen:{req_id}]   {'t2v' if is_video else 't2i'} inference {_ms(t_inf)}", flush=True)
                return result

            # Do NOT use `with ThreadPoolExecutor(...) as _ex:` — its __exit__
            # calls shutdown(wait=True) which blocks until the inference thread
            # finishes even when a TimeoutError has already been raised, causing
            # the HTTP 504 response to be delayed until inference completes
            # (which can be 10+ min on MPS, longer than the bridge timeout).
            _ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            future = _ex.submit(_run_inference)
            _ex.shutdown(wait=False)  # detach — thread runs to completion in bg
            try:
                media = future.result(timeout=_INFERENCE_TIMEOUT)
            except concurrent.futures.TimeoutError:
                print(
                    f"[gen:{req_id}] ✗ inference timeout after {_INFERENCE_TIMEOUT}s", flush=True)
                raise HTTPException(
                    status_code=504, detail=f"Inference timed out after {_INFERENCE_TIMEOUT}s")
        except HTTPException:
            raise
        except Exception as e:
            print(
                f"[gen:{req_id}] ✗ inference error ({mode}) after {_ms(t_req)}: {e}", flush=True)
            traceback.print_exc()
            raise HTTPException(
                status_code=500, detail=f"Inference error ({mode}): {e}")

    t_enc = time.time()
    meta = {
        "model": req.model,
        "seed": req.seed,
        "steps": effective_steps,
        "width": req.width,
        "height": req.height,
        "guidance_scale": round(_effective_guidance(req.guidance_scale, adapter.default_guidance), 2),
    }
    elapsed = round(time.time() - t_req, 2)

    if is_video:
        meta["num_frames"] = req.num_frames
        meta["fps"] = req.fps
        video_url = _encode_video(media, req.fps)
        print(f"[gen:{req_id}]   encode video {_ms(t_enc)} | frames={len(media)}", flush=True)
        print(f"[gen:{req_id}] ✓ done | mode={mode} total={elapsed}s seed={req.seed}", flush=True)
        return {
            "videoUrl": video_url,
            "mode": mode,
            "elapsed": elapsed,
            "meta": meta,
        }
    else:
        image_url = _encode_image(media)
        print(f"[gen:{req_id}]   encode {_ms(t_enc)} | size={media.size}", flush=True)
        print(f"[gen:{req_id}] ✓ done | mode={mode} total={elapsed}s seed={req.seed}", flush=True)
        return {
            "imageUrl": image_url,
            "mode": mode,
            "elapsed": elapsed,
            "meta": meta,
        }


@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest):
    if not req.model:
        raise HTTPException(status_code=400, detail="model is required")
    try:
        return service.generate_chat_completion(req)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
