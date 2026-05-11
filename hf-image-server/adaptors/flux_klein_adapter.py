"""
FLUX.2 Klein adapter for the image generation server.

Supports:
  black-forest-labs/FLUX.2-klein-9B — 9B step-distilled, unified t2i + i2i,
                                       4 steps, guidance_scale=1.0, bfloat16
                                       non-commercial license

The Flux2KleinPipeline handles both text-to-image and image-to-image in a
single unified architecture; passing `image=` switches to editing mode.
"""

import torch
from PIL import Image as PILImage

from .model_adapter import ModelAdapter
from .utils import _apply_pipeline_optimizations, _call_with_progress, _effective_steps, _effective_guidance, DEVICE


class FluxKleinAdapter(ModelAdapter):
    """
    Adapter for the black-forest-labs FLUX.2 Klein family (9B).

    Distilled to 4 steps with guidance_scale=1.0.
    Unified model: same pipeline for t2i and i2i.
    """

    default_steps = 4
    default_guidance = 1.0
    supports_i2i = True

    @classmethod
    def matches(cls, model_id: str) -> bool:
        lower = model_id.lower()
        # Exclude the -kv variant — handled by FluxKleinKVAdapter
        if "klein" in lower and "kv" in lower:
            return False
        return "flux.2" in lower or ("flux" in lower and "klein" in lower)

    def load(self, model_id: str) -> tuple:
        from diffusers import Flux2KleinPipeline

        pipe = Flux2KleinPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)

        if DEVICE == "cuda":
            # cpu_offload keeps the ~29 GB model manageable even on 24 GB VRAM
            pipe.enable_model_cpu_offload()
        elif DEVICE == "mps":
            # 9B in bfloat16 ≈ 18 GB — cpu_offload avoids exhausting unified memory
            pipe.enable_model_cpu_offload()
        else:
            pipe = pipe.to(DEVICE)

        _apply_pipeline_optimizations(pipe)
        return pipe, pipe  # same pipeline for both t2i and i2i

    def text_to_image(self, pipes: tuple, req) -> PILImage.Image:
        pipe, _ = pipes
        generator = torch.Generator("cpu").manual_seed(req.seed) if req.seed is not None else None

        return _call_with_progress(
            pipe,
            prompt=req.prompt,
            height=req.height,
            width=req.width,
            num_inference_steps=_effective_steps(req.num_inference_steps, self.default_steps),
            guidance_scale=_effective_guidance(req.guidance_scale, self.default_guidance),
            generator=generator,
        ).images[0]

    def image_to_image(self, pipes: tuple, req, ref_img: PILImage.Image) -> PILImage.Image:
        pipe, _ = pipes

        # Reference image must match the target output dimensions to avoid
        # shape errors in the conditioning encoder.
        if ref_img.size != (req.width, req.height):
            ref_img = ref_img.resize((req.width, req.height), PILImage.LANCZOS)

        generator = torch.Generator("cpu").manual_seed(req.seed) if req.seed is not None else None

        return _call_with_progress(
            pipe,
            prompt=req.prompt,
            image=ref_img,
            height=req.height,
            width=req.width,
            num_inference_steps=_effective_steps(req.num_inference_steps, self.default_steps),
            guidance_scale=_effective_guidance(req.guidance_scale, self.default_guidance),
            generator=generator,
        ).images[0]
