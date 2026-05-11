"""
FLUX adapter for the image generation server.

Supports:
  black-forest-labs/FLUX.1-schnell  — 4-step distilled, guidance_scale=0.0, Apache-2.0
  black-forest-labs/FLUX.1-dev      — 20-step guided,   guidance_scale=3.5, non-commercial
"""

import torch
from PIL import Image as PILImage

from .model_adapter import ModelAdapter
from .utils import _apply_pipeline_optimizations, _call_with_progress, DEVICE

# FLUX transformers are trained in bfloat16; float16 can produce NaNs.
_FLUX_DTYPE = torch.bfloat16


def _is_schnell(model_id: str) -> bool:
    return "schnell" in model_id.lower()


class FluxAdapter(ModelAdapter):
    """
    Adapter for the black-forest-labs FLUX.1 family.

    Schnell: guidance-distilled, 4 steps, guidance_scale=0.0, max_sequence_length=256
    Dev:     guidance-trained,   20 steps, guidance_scale=3.5, max_sequence_length=512

    Text-to-image only — the base FluxPipeline does not expose img2img.
    """

    default_steps = 4       # schnell default; dev overrides per-call
    default_guidance = 0.0  # schnell default; dev overrides per-call
    supports_i2i = False

    @classmethod
    def matches(cls, model_id: str) -> bool:
        lower = model_id.lower()
        # Matches black-forest-labs/FLUX.1-* and community forks that keep "flux" in the id
        return "flux" in lower and (
            "black-forest-labs" in lower
            or lower.startswith("flux")
            or "/flux" in lower
        )

    def load(self, model_id: str) -> tuple:
        from diffusers import FluxPipeline

        pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=_FLUX_DTYPE)

        if DEVICE == "cuda":
            # On CUDA let diffusers manage device placement via cpu_offload so
            # multiple models can coexist without OOM.
            pipe.enable_model_cpu_offload()
        elif DEVICE == "mps":
            # FLUX.1 is ~24 GB in bfloat16. cpu_offload keeps weights on CPU
            # and moves each sub-module to MPS only during its forward pass,
            # avoiding a one-shot 24 GB transfer that would exhaust unified memory.
            pipe.enable_model_cpu_offload()
        else:
            pipe = pipe.to(DEVICE)

        _apply_pipeline_optimizations(pipe)
        return pipe, None  # no separate i2i pipeline

    def text_to_image(self, pipes: tuple, req) -> PILImage.Image:
        pipe, _ = pipes
        schnell = _is_schnell(req.model)

        # Resolve per-variant defaults when sentinels are present
        steps    = req.num_inference_steps if req.num_inference_steps != 4 else (4 if schnell else 20)
        guidance = req.guidance_scale      if req.guidance_scale      != 0.0 else (0.0 if schnell else 3.5)
        max_seq  = 256 if schnell else 512

        generator = torch.Generator("cpu").manual_seed(req.seed) if req.seed is not None else None

        return _call_with_progress(
            pipe,
            prompt=req.prompt,
            width=req.width,
            height=req.height,
            num_inference_steps=steps,
            guidance_scale=guidance,
            max_sequence_length=max_seq,
            generator=generator,
        ).images[0]
