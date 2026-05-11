"""
Catch-all GGUF adapter.

Tries StableDiffusionXLPipeline.from_single_file() first (handles most
community .gguf / .safetensors dumps), then falls back to the base SD
pipeline.  This covers SDXL, SD 1.5, Illustrious, PonyXL, etc.
"""

import os
import torch
from PIL import Image as PILImage

from .model_adapter import ModelAdapter
from .utils import _apply_pipeline_optimizations, _call_with_progress, _effective_steps, _effective_guidance


class GGUFGenericAdapter(ModelAdapter):
    default_steps = 20
    default_guidance = 7.0
    supports_i2i = False

    @classmethod
    def matches(cls, model_path: str) -> bool:
        return model_path.endswith(".gguf")

    def load(self, model_path: str) -> tuple:
        print(f"[lms] GGUFGenericAdapter loading {model_path}", flush=True)
        from diffusers import GGUFQuantizationConfig

        quant_config = GGUFQuantizationConfig(compute_dtype=torch.float16)

        # Try SDXL first (most common GGUF format in the community)
        try:
            from diffusers import StableDiffusionXLPipeline
            pipe = StableDiffusionXLPipeline.from_single_file(
                model_path,
                quantization_config=quant_config,
                torch_dtype=torch.float16,
                use_safetensors=False,
            )
            pipe.enable_model_cpu_offload()
            _apply_pipeline_optimizations(pipe)
            return (pipe, None)
        except Exception as e:
            print(f"[lms] SDXL load failed ({e}), trying SD …", flush=True)

        from diffusers import StableDiffusionPipeline
        pipe = StableDiffusionPipeline.from_single_file(
            model_path,
            quantization_config=quant_config,
            torch_dtype=torch.float16,
        )
        pipe.enable_model_cpu_offload()
        _apply_pipeline_optimizations(pipe)
        return (pipe, None)

    def text_to_image(self, pipes: tuple, req) -> PILImage.Image:
        pipe, _ = pipes
        steps = _effective_steps(req.num_inference_steps, self.default_steps)
        guidance = _effective_guidance(req.guidance_scale, self.default_guidance)
        generator = torch.Generator().manual_seed(req.seed)
        result = _call_with_progress(
            pipe,
            prompt=req.prompt,
            num_inference_steps=steps,
            guidance_scale=guidance,
            width=req.width,
            height=req.height,
            generator=generator,
        )
        return result.images[0]
