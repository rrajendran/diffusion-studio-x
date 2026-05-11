"""
FastAPI routes for the lms-image-server.

Endpoints
---------
GET  /health             — liveness check
GET  /models             — list GGUF/MLX models from ~/.lmstudio/models
GET  /model-capabilities — per-family feature flags
GET  /progress           — current inference step (for progress bars)
POST /preload            — background-load a model (non-blocking)
POST /generate           — run text-to-image (blocking)
POST /unload             — release the loaded model from memory
"""

import concurrent.futures
import os
import time
import traceback
import threading
import urllib.request
import json

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from adaptors import (
    _effective_steps,
    _effective_guidance,
    _decode_image,
    _encode_image,
    _progress,
    get_adapter,
)
from domain import GenerateRequest, PreloadRequest
from scanner import scan_models, _looks_like_image_model
from services import service

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_INFERENCE_TIMEOUT = int(os.environ.get("INFERENCE_TIMEOUT", "1200"))
_LMS_MODELS_DIR = os.environ.get("LMS_MODELS_DIR", "")
_LM_STUDIO_BASE = os.environ.get("LM_STUDIO_BASE_URL", "http://localhost:1234").rstrip("/")


def _fetch_lmstudio_models() -> list[dict]:
    """Fetch models from the running LM Studio app via its REST API."""
    try:
        req = urllib.request.Request(
            f"{_LM_STUDIO_BASE}/api/v1/models",
            headers={"Accept": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=3) as resp:  # noqa: S310
            data = json.loads(resp.read())
        return [
            {
                "key":          m["key"],
                "relative_key": m["key"],
                "display_name": m.get("display_name") or m["key"],
                "format":       m.get("type", "llm"),
                "size_bytes":   m.get("size_bytes", 0),
                "source":       "lmstudio",
                "loaded":       len(m.get("loaded_instances", [])) > 0,
                "loaded_instances": m.get("loaded_instances", []),
            }
            for m in data.get("models", [])
            if _looks_like_image_model(m.get("key", "") + " " + m.get("display_name", ""))
        ]
    except Exception as e:
        print(f"[lms] LM Studio API unavailable: {e}", flush=True)
        return []


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": service.get_cached_model_path(),
    }


@app.get("/models")
def list_models():
    """
    Return all image-gen models from two sources:
      1. Local GGUF/MLX files scanned from ~/.lmstudio/models  (source='local')
      2. Models reported by the running LM Studio app API       (source='lmstudio')
    Deduplicated by key; local entries win on conflict.
    """
    local = scan_models(_LMS_MODELS_DIR or None)
    for m in local:
        m.setdefault("source", "local")
        m.setdefault("loaded", False)

    lmstudio = _fetch_lmstudio_models()

    # Merge: local entries take priority; append LM Studio entries not already present
    local_keys = {m["key"] for m in local}
    merged = local + [m for m in lmstudio if m["key"] not in local_keys]
    return merged


@app.get("/model-capabilities")
def model_capabilities():
    """Return capabilities for each supported model family."""
    from adaptors import (
        GGUFZImageAdapter, GGUFFluxAdapter, GGUFSD3Adapter,
        MLXFluxAdapter, GGUFGenericAdapter,
    )
    return [
        {"family": "Z-Image-GGUF",  "pattern": "z-image",         **GGUFZImageAdapter().capabilities()},
        {"family": "FLUX-GGUF",     "pattern": "flux",             **GGUFFluxAdapter().capabilities()},
        {"family": "SD3-GGUF",      "pattern": "sd3",              **GGUFSD3Adapter().capabilities()},
        {"family": "FLUX-MLX",      "pattern": "flux (mlx dir)",   **MLXFluxAdapter().capabilities()},
        {"family": "Generic-GGUF",  "pattern": "*",                **GGUFGenericAdapter().capabilities()},
    ]


@app.get("/progress")
def get_progress():
    return _progress


@app.post("/preload")
def preload(req: PreloadRequest):
    """Kick off background model load; returns immediately."""
    def _load():
        try:
            service.get_pipes(req.model)
            print(f"[lms] preload complete: {req.model}", flush=True)
        except Exception as e:
            print(f"[lms] preload failed: {e}", flush=True)

    threading.Thread(target=_load, daemon=True).start()
    return {"status": "loading", "model": req.model}


@app.post("/unload")
def unload():
    """Unload the currently loaded model and free memory."""
    freed = service.unload()
    return {"status": "unloaded", "model": freed}


@app.post("/generate")
def generate(req: GenerateRequest):
    if not req.model:
        raise HTTPException(status_code=400, detail="model is required")

    _progress.update({"step": 0, "total": 0})

    t_req = time.time()
    req_id = int(t_req * 1000) % 1_000_000
    adapter = get_adapter(req.model)
    effective_steps = _effective_steps(req.num_inference_steps, adapter.default_steps)

    print(
        f"[gen:{req_id}] ▶ {os.path.basename(req.model)} "
        f"{req.width}x{req.height} steps={effective_steps} "
        f"adapter={type(adapter).__name__}",
        flush=True,
    )

    # Load (or hit cache)
    t_load = time.time()
    cache_hit = service.get_cached_model_path() == req.model
    try:
        pipes = service.get_pipes(req.model)
    except Exception as e:
        print(f"[gen:{req_id}] ✗ load failed ({_ms(t_load)}): {e}", flush=True)
        raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")

    print(f"[gen:{req_id}]   load {'(cached)' if cache_hit else '(cold)'} {_ms(t_load)}", flush=True)
    _progress.update({"step": 0, "total": effective_steps})

    mode = "t2i"

    with service.get_inference_lock():
        try:
            def _run():
                nonlocal mode
                with torch.inference_mode():
                    if req.reference_image and adapter.supports_i2i:
                        mode = "i2i"
                        ref_img = _decode_image(req.reference_image)
                        return adapter.image_to_image(pipes, req, ref_img)
                    else:
                        if req.reference_image:
                            print(f"[gen:{req_id}]   i2i not supported → t2i fallback", flush=True)
                        return adapter.text_to_image(pipes, req)

            _ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            future = _ex.submit(_run)
            _ex.shutdown(wait=False)
            try:
                image = future.result(timeout=_INFERENCE_TIMEOUT)
            except concurrent.futures.TimeoutError:
                raise HTTPException(
                    status_code=504,
                    detail=f"Inference timed out after {_INFERENCE_TIMEOUT}s",
                )
        except HTTPException:
            raise
        except Exception as e:
            print(f"[gen:{req_id}] ✗ inference error: {e}", flush=True)
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    t_enc = time.time()
    image_url = _encode_image(image)
    print(f"[gen:{req_id}]   encode {_ms(t_enc)}", flush=True)

    elapsed = round(time.time() - t_req, 2)
    print(f"[gen:{req_id}] ✓ done | mode={mode} total={elapsed}s", flush=True)

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
            "guidance_scale": round(
                _effective_guidance(req.guidance_scale, adapter.default_guidance), 2
            ),
        },
    }


def _ms(t: float) -> str:
    return f"{(time.time() - t) * 1000:.0f}ms"
