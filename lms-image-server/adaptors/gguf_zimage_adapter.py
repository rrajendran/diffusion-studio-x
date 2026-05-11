"""
Adapter for Z-Image-Turbo in GGUF format (e.g. unsloth/Z-Image-Turbo-GGUF).

Loading strategy
----------------
diffusers >= 0.33 supports loading GGUF-quantised transformers via
GGUFQuantizationConfig + from_single_file().  ZImagePipeline uses a
custom UNet architecture; we load the quantised weights from the .gguf
file and build the pipeline around them.

If the ZImagePipeline cannot locate the VAE / text-encoder from the
GGUF bundle alone, we fall back to loading those components from the
HF-cached `Tongyi-MAI/Z-Image-Turbo` snapshot.
"""

import os
import torch
from PIL import Image as PILImage

from .model_adapter import ModelAdapter
from .utils import DEVICE, _apply_pipeline_optimizations, _call_with_progress, _effective_steps, _effective_guidance


class GGUFZImageAdapter(ModelAdapter):
    default_steps = 9
    default_guidance = 0.0
    supports_i2i = False

    @classmethod
    def matches(cls, model_path: str) -> bool:
        lower = os.path.basename(model_path).lower()
        return model_path.endswith(".gguf") and ("z-image" in lower or "zimage" in lower)

    def load(self, model_path: str) -> tuple:
        from diffusers import ZImagePipeline, GGUFQuantizationConfig

        print(f"[lms] GGUFZImageAdapter loading {model_path}", flush=True)
        quant_config = GGUFQuantizationConfig(compute_dtype=torch.bfloat16)

        try:
            pipe = ZImagePipeline.from_single_file(
                model_path,
                quantization_config=quant_config,
                torch_dtype=torch.bfloat16,
            )
        except Exception as e:
            # Fallback: load GGUF transformer + remaining components from HF cache
            print(f"[lms] from_single_file failed ({e}), trying transformer-only load …", flush=True)
            from diffusers import ZImageTransformer2DModel

            transformer = ZImageTransformer2DModel.from_single_file(
                model_path,
                quantization_config=quant_config,
                torch_dtype=torch.bfloat16,
            )
            pipe = ZImagePipeline.from_pretrained(
                "Tongyi-MAI/Z-Image-Turbo",
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
