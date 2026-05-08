"""
Services for the image generation server.
"""

import gc
import os
import threading
import time

import torch

from adaptors import (
    DEVICE,
    get_adapter,
    ModelAdapter,
)


class ImageGenerationService:
    """Service class for managing image generation pipelines and operations."""

    def __init__(self):
        self._cached_model_id: str | None = None
        self._cached_pipes: tuple | None = None
        self._inference_lock = threading.Lock()
        self._loading_lock = threading.Lock()   # serialises model loads
        self._chat_model_id: str | None = None
        self._chat_pipeline = None
        self._chat_lock = threading.Lock()

    def _detect_model_source(self, model_id: str) -> str:
        """Return a short label describing where the model will be loaded from."""
        if os.path.isdir(model_id):
            return f"local path ({model_id})"
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(model_id, local_files_only=True)
            return "local HF cache"
        except Exception:
            return "HuggingFace Hub (download)"

    def _warmup_pipeline(self, pipes: tuple, adapter: ModelAdapter) -> None:
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

        # Acquire inference lock non-blocking: if real inference is already
        # running, skip warmup rather than race against it on shared pipeline
        # components. Holding the lock prevents concurrent inference during warmup.
        if not self._inference_lock.acquire(blocking=False):
            print("[hf-image-server] warmup skipped: inference in progress", flush=True)
            return
        t0 = time.time()
        print("[hf-image-server] warming up …", flush=True)
        try:
            with torch.inference_mode():
                adapter.text_to_image(pipes, _Req())
            if DEVICE == "mps":
                torch.mps.synchronize()
            print(f"[hf-image-server] warmup done ({time.time() - t0:.1f}s)", flush=True)
        except Exception as e:
            print(f"[hf-image-server] warmup skipped: {e}", flush=True)
        finally:
            self._inference_lock.release()

    def get_pipes(self, model_id: str) -> tuple:
        """Get cached or load new pipelines for the given model."""
        # Fast path: no lock needed when the right model is already cached.
        if self._cached_pipes is not None and self._cached_model_id == model_id:
            return self._cached_pipes

        # Slow path: serialise all loaders so only one thread loads at a time.
        # Double-check after acquiring the lock — a concurrent caller may have
        # finished loading by the time we get here.
        with self._loading_lock:
            if self._cached_pipes is not None and self._cached_model_id == model_id:
                return self._cached_pipes

            if self._cached_pipes is not None:
                print(f"[hf-image-server] unloading {self._cached_model_id} …", flush=True)
                del self._cached_pipes
                self._cached_pipes = None
                gc.collect()
                if DEVICE == "cuda":
                    torch.cuda.empty_cache()
                elif DEVICE == "mps":
                    torch.mps.empty_cache()
                # Force garbage collection multiple times for better cleanup
                for _ in range(3):
                    gc.collect()

            source = self._detect_model_source(model_id)
            print(f"[hf-image-server] loading {model_id} | source={source}", flush=True)

            # Check available memory before loading
            import psutil
            memory = psutil.virtual_memory()
            print(f"[hf-image-server] system memory: {memory.available / 1024**3:.1f}GB available", flush=True)

            t0 = time.time()

            try:
                adapter = get_adapter(model_id)
                print(f"[hf-image-server] adapter loaded: {adapter.__class__.__name__}", flush=True)
                self._cached_pipes = adapter.load(model_id)
                print(f"[hf-image-server] model loaded successfully", flush=True)
                self._cached_model_id = model_id
                print(f"[hf-image-server] ready ({time.time() - t0:.1f}s) | source={source}", flush=True)

                # Try to warmup the pipeline, but don't fail if it doesn't work
                try:
                    self._warmup_pipeline(self._cached_pipes, adapter)
                except Exception as warmup_e:
                    print(f"[hf-image-server] warmup failed, continuing anyway: {warmup_e}", flush=True)
            except Exception as e:
                print(f"[hf-image-server] failed to load {model_id}: {e}", flush=True)
                # Clean up any partial state
                self._cached_pipes = None
                self._cached_model_id = None
                raise

            return self._cached_pipes

    def get_inference_lock(self):
        """Get the inference lock for serializing requests."""
        return self._inference_lock

    def get_cached_model_id(self):
        """Get the currently cached model ID."""
        return self._cached_model_id

    def _get_chat_pipeline(self, model_id: str):
        """Get or load the chat pipeline for text generation."""
        if self._chat_model_id == model_id and self._chat_pipeline is not None:
            return self._chat_pipeline
        print(f"[chat] loading text model {model_id!r} ...", flush=True)
        from transformers import pipeline as hf_pipeline
        from adaptors import DTYPE
        self._chat_pipeline = hf_pipeline(
            "text-generation",
            model=model_id,
            device_map="auto",
            dtype=DTYPE,
        )
        self._chat_model_id = model_id
        print(f"[chat] model ready", flush=True)
        return self._chat_pipeline

    def generate_chat_completion(self, req):
        """Generate a chat completion response."""
        t0 = time.time()
        try:
            with self._chat_lock:
                pipe = self._get_chat_pipeline(req.model)
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
            import traceback
            traceback.print_exc()
            raise Exception(f"Chat completion error: {e}")


# Global service instance
service = ImageGenerationService()