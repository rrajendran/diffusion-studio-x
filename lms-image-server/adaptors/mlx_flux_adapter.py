"""
Adapter for FLUX models in MLX format (Apple Silicon only).

Requires:  pip install mflux

MLX FLUX models are stored as directories containing safetensors weights
and a config file.  The `mflux` library handles quantisation and inference;
no HuggingFace download is needed at inference time (weights are local).

Model variants
--------------
  "schnell"  → 4-step, no guidance  (fast, uncensored)
  "dev"      → ~20 steps, cfg 3.5   (higher quality)
Detected from the directory name; defaults to schnell if ambiguous.
"""

import os
import torch
from PIL import Image as PILImage

from .model_adapter import ModelAdapter
from .utils import _effective_steps, _effective_guidance


def _is_schnell(path: str) -> bool:
    return "schnell" in os.path.basename(path).lower()


class MLXFluxAdapter(ModelAdapter):
    supports_i2i = False

    @property
    def default_steps(self):
        return 4

    @property
    def default_guidance(self):
        return 0.0

    @classmethod
    def matches(cls, model_path: str) -> bool:
        lower = os.path.basename(model_path).lower()
        # Directory-based MLX model with flux in name
        return os.path.isdir(model_path) and "flux" in lower

    def load(self, model_path: str) -> tuple:
        try:
            from mflux import Flux1
        except ImportError:
            raise RuntimeError(
                "mflux not installed. Run: pip install mflux\n"
                "mflux is required for MLX-format FLUX models on Apple Silicon."
            )

        print(f"[lms] MLXFluxAdapter loading {model_path}", flush=True)
        # mflux Flux1.from_path expects a local directory containing MLX weights
        flux = Flux1.from_path(model_path)
        return (flux, None)

    def text_to_image(self, pipes: tuple, req) -> PILImage.Image:
        from mflux import Config

        flux, _ = pipes
        is_schnell = _is_schnell(req.model)
        steps = _effective_steps(req.num_inference_steps, 4 if is_schnell else 20)
        guidance = _effective_guidance(req.guidance_scale, 0.0 if is_schnell else 3.5)

        config = Config(
            num_inference_steps=steps,
            height=req.height,
            width=req.width,
            guidance=guidance,
        )
        result = flux.generate_image(
            seed=req.seed,
            prompt=req.prompt,
            config=config,
        )
        # mflux returns an object with a .image PIL attribute
        img = result.image if hasattr(result, "image") else result
        if not isinstance(img, PILImage.Image):
            img = PILImage.fromarray(img)
        return img
