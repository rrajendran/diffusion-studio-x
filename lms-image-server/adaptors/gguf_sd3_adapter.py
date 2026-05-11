"""
Adapter for Stable Diffusion 3 (SD3) models in GGUF format.
"""

import os
import torch
from PIL import Image as PILImage

from .model_adapter import ModelAdapter
from .utils import _apply_pipeline_optimizations, _call_with_progress, _effective_steps, _effective_guidance


class GGUFSD3Adapter(ModelAdapter):
    default_steps = 28
    default_guidance = 7.0
    supports_i2i = False

    @classmethod
    def matches(cls, model_path: str) -> bool:
        lower = os.path.basename(model_path).lower()
        return model_path.endswith(".gguf") and (
            "sd3" in lower or "stable-diffusion-3" in lower or "stable_diffusion_3" in lower
        )

    def load(self, model_path: str) -> tuple:
        from diffusers import (
            StableDiffusion3Pipeline,
            SD3Transformer2DModel,
            GGUFQuantizationConfig,
        )

        print(f"[lms] GGUFSD3Adapter loading {model_path}", flush=True)
        quant_config = GGUFQuantizationConfig(compute_dtype=torch.bfloat16)

        try:
            pipe = StableDiffusion3Pipeline.from_single_file(
                model_path,
                quantization_config=quant_config,
                torch_dtype=torch.bfloat16,
            )
        except Exception as e:
            print(f"[lms] from_single_file failed ({e}), trying transformer-only …", flush=True)
            transformer = SD3Transformer2DModel.from_single_file(
                model_path,
                quantization_config=quant_config,
                torch_dtype=torch.bfloat16,
            )
            pipe = StableDiffusion3Pipeline.from_pretrained(
                "stabilityai/stable-diffusion-3-medium-diffusers",
                transformer=transformer,
                torch_dtype=torch.bfloat16,
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
