"""
Model service — load, cache, unload, and run inference.

Key differences from hf-image-server/services.py:
  • Models are loaded from local GGUF/MLX files (no HF download in the hot path)
  • Explicit POST /unload endpoint: frees VRAM / RAM on demand
  • No chat-completion support (text LLMs are handled by LM Studio itself)
"""

import gc
import threading
import time

import torch

from adaptors import DEVICE, get_adapter, ModelAdapter


class LmsImageService:

    def __init__(self):
        self._cached_model_path: str | None = None
        self._cached_pipes: tuple | None = None
        self._inference_lock = threading.Lock()
        self._loading_lock = threading.Lock()

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _free_cache(self) -> None:
        """Release the current pipeline and reclaim memory."""
        if self._cached_pipes is None:
            return
        print(f"[lms] unloading {self._cached_model_path} …", flush=True)
        del self._cached_pipes
        self._cached_pipes = None
        self._cached_model_path = None
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        elif DEVICE == "mps":
            torch.mps.empty_cache()
        for _ in range(3):
            gc.collect()

    def _warmup(self, pipes: tuple, adapter: ModelAdapter) -> None:
        if DEVICE not in ("mps", "cuda"):
            return

        class _Req:
            prompt = "warmup"
            width = 512
            height = 512
            num_inference_steps = 2
            guidance_scale = adapter.default_guidance
            seed = 0
            model = ""

        if not self._inference_lock.acquire(blocking=False):
            return
        t0 = time.time()
        print("[lms] warming up …", flush=True)
        try:
            with torch.inference_mode():
                adapter.text_to_image(pipes, _Req())
            if DEVICE == "mps":
                torch.mps.synchronize()
            print(f"[lms] warmup done ({time.time() - t0:.1f}s)", flush=True)
        except Exception as e:
            print(f"[lms] warmup skipped: {e}", flush=True)
        finally:
            self._inference_lock.release()

    # ── Public API ─────────────────────────────────────────────────────────────

    def get_pipes(self, model_path: str) -> tuple:
        """Return cached pipeline or load from disk."""
        # Fast path
        if self._cached_pipes is not None and self._cached_model_path == model_path:
            return self._cached_pipes

        with self._loading_lock:
            if self._cached_pipes is not None and self._cached_model_path == model_path:
                return self._cached_pipes

            self._free_cache()

            print(f"[lms] loading {model_path}", flush=True)
            t0 = time.time()
            adapter = get_adapter(model_path)
            print(f"[lms] adapter: {adapter.__class__.__name__}", flush=True)

            self._cached_pipes = adapter.load(model_path)
            self._cached_model_path = model_path
            print(f"[lms] ready ({time.time() - t0:.1f}s)", flush=True)

            try:
                self._warmup(self._cached_pipes, adapter)
            except Exception as e:
                print(f"[lms] warmup error (non-fatal): {e}", flush=True)

            return self._cached_pipes

    def unload(self) -> str | None:
        """Explicitly unload the current model; returns the model path that was freed."""
        with self._loading_lock:
            path = self._cached_model_path
            self._free_cache()
            return path

    def get_inference_lock(self):
        return self._inference_lock

    def get_cached_model_path(self) -> str | None:
        return self._cached_model_path


service = LmsImageService()
