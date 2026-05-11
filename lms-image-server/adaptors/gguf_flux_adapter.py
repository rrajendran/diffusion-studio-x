"""
Adapter for FLUX.1 models in GGUF format.

GGUF FLUX files from the community (e.g. city96/FLUX.1-schnell-gguf)
contain only the transformer weights.  We load the transformer from
the GGUF file and fill the rest of the pipeline (VAE, text encoders)
from the local HF cache.

Schnell (distilled, 4-step) vs Dev (guidance-distilled, ~20-step) is
detected from the filename: "schnell" → default 4 steps / cfg 0.0,
anything else → 20 steps / cfg 3.5.
"""

import os
import torch
from PIL import Image as PILImage

from .model_adapter import ModelAdapter
from .utils import DEVICE, _apply_pipeline_optimizations, _call_with_progress, _effective_steps, _effective_guidance


def _is_schnell(path: str) -> bool:
    return "schnell" in os.path.basename(path).lower()


class GGUFFluxAdapter(ModelAdapter):
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
        return model_path.endswith(".gguf") and "flux" in lower

    def load(self, model_path: str) -> tuple:
        from diffusers import (
            FluxPipeline,
            FluxTransformer2DModel,
            GGUFQuantizationConfig,
        )

        is_schnell = _is_schnell(model_path)
        hf_repo = "black-forest-labs/FLUX.1-schnell" if is_schnell else "black-forest-labs/FLUX.1-dev"

        print(f"[lms] GGUFFluxAdapter loading {model_path} (repo={hf_repo})", flush=True)
        quant_config = GGUFQuantizationConfig(compute_dtype=torch.bfloat16)

        transformer = FluxTransformer2DModel.from_single_file(
            model_path,
            quantization_config=quant_config,
            torch_dtype=torch.bfloat16,
        )
        pipe = FluxPipeline.from_pretrained(
            hf_repo,
            transformer=transformer,
            torch_dtype=torch.bfloat16,
        )
        pipe.enable_model_cpu_offload()
        _apply_pipeline_optimizations(pipe)
        return (pipe, None)

    def text_to_image(self, pipes: tuple, req) -> PILImage.Image:
        pipe, _ = pipes
        is_schnell = _is_schnell(req.model)
        steps = _effective_steps(req.num_inference_steps, 4 if is_schnell else 20)
        guidance = _effective_guidance(req.guidance_scale, 0.0 if is_schnell else 3.5)
        generator = torch.Generator().manual_seed(req.seed)
        kwargs = dict(
            prompt=req.prompt,
            num_inference_steps=steps,
            width=req.width,
            height=req.height,
            generator=generator,
        )
        if not is_schnell:
            kwargs["guidance_scale"] = guidance
        result = _call_with_progress(pipe, **kwargs)
        return result.images[0]
