"""
Wan2.2 adapter for text-to-video (T2V) and image-to-video (I2V) generation.

Supports:
  Wan-AI/Wan2.2-T2V-A14B-Diffusers  — Text-to-video, 14B params
  Wan-AI/Wan2.2-I2V-A14B-Diffusers  — Image-to-video, 14B params
"""

import torch
from PIL import Image as PILImage

from .model_adapter import ModelAdapter
from .utils import _apply_pipeline_optimizations, _call_with_progress, DEVICE


# Wan2.2 native 480p: max ~400 K pixels, each dimension multiple of 16.
_WAN_MAX_PIXELS = 832 * 480

# At 480p + N frames, the self-attention sequence length is:
#   T_lat × H_lat × W_lat  where T_lat = (N-1)//4 + 1, H_lat = H//8, W_lat = W//8
# Q×K bytes = seq_len² × 2.  For safe MPS (≤ ~2 GB per head), we need seq_len ≤ ~32K.
# 17 frames → T_lat = 5 → seq_len = 5 × 60 × 104 = 31,200 ✓
# 49 frames → T_lat = 13 → seq_len = 13 × 60 × 104 = 81,120 → 13.2 GB ✗
_WAN_MAX_FRAMES_MPS = 17


def _is_i2v(model_id: str) -> bool:
    return "i2v" in model_id.lower()


def _wan_resolution(width: int, height: int) -> tuple[int, int]:
    """Clamp to Wan2.2 native 480p to keep the attention buffer within MPS limits."""
    if width * height <= _WAN_MAX_PIXELS:
        # Already within 480p — still snap to multiples of 16
        w = max(16, round(width / 16) * 16)
        h = max(16, round(height / 16) * 16)
        return w, h
    ratio = width / height
    h = int((_WAN_MAX_PIXELS / ratio) ** 0.5)
    w = int(h * ratio)
    w = max(16, round(w / 16) * 16)
    h = max(16, round(h / 16) * 16)
    return w, h


def _wan_frames(num_frames: int) -> int:
    """On MPS, clamp to the safe maximum that keeps attention within ~2 GB per head."""
    if DEVICE == "mps" and num_frames > _WAN_MAX_FRAMES_MPS:
        print(f"[wan] clamping num_frames {num_frames} → {_WAN_MAX_FRAMES_MPS} (MPS memory limit)", flush=True)
        return _WAN_MAX_FRAMES_MPS
    return num_frames


# Wan2.2 is bfloat16 on CUDA; MPS bfloat16 has incomplete kernel coverage → use float16.
def _wan_dtype() -> torch.dtype:
    if DEVICE == "mps":
        return torch.float16
    return torch.bfloat16


class WanAdapter(ModelAdapter):
    """
    Adapter for Wan-AI Wan2.2 video generation models.

    T2V: text prompt → video frames
    I2V: reference image + text prompt → video frames (uses supports_i2i flag)

    Step budget on MPS (14B, cpu_offload): ~40-50s/step.
    default_steps=5 → ~4 min; use_steps override to increase quality if time allows.
    """

    default_steps = 5      # fast default for MPS; override in settings for better quality
    default_guidance = 5.0
    supports_i2i = True   # I2V model handles reference images
    is_video = True

    @classmethod
    def matches(cls, model_id: str) -> bool:
        lower = model_id.lower()
        return "wan" in lower and "wan-ai" in lower

    def load(self, model_id: str) -> tuple:
        dtype = _wan_dtype()
        if _is_i2v(model_id):
            from diffusers import WanImageToVideoPipeline
            pipe = WanImageToVideoPipeline.from_pretrained(
                model_id, torch_dtype=dtype
            )
        else:
            from diffusers import WanPipeline
            pipe = WanPipeline.from_pretrained(
                model_id, torch_dtype=dtype
            )

        if DEVICE in ("cuda", "mps"):
            pipe.enable_model_cpu_offload()
        else:
            pipe = pipe.to(DEVICE)

        _apply_pipeline_optimizations(pipe)
        return pipe, None

    def text_to_image(self, pipes: tuple, req) -> PILImage.Image:
        # Satisfy abstract requirement — callers should use text_to_video for this adapter.
        frames = self.text_to_video(pipes, req)
        return frames[0] if frames else PILImage.new("RGB", (req.width, req.height))

    def text_to_video(self, pipes: tuple, req) -> list:
        if _is_i2v(req.model):
            raise ValueError(
                "Wan2.2-I2V requires a reference image. Attach an image before generating, "
                "or switch to Wan2.2-T2V for text-only video generation."
            )
        pipe, _ = pipes
        steps = req.num_inference_steps if req.num_inference_steps != 4 else self.default_steps
        guidance = req.guidance_scale if req.guidance_scale != 0.0 else self.default_guidance
        width, height = _wan_resolution(req.width, req.height)
        num_frames = _wan_frames(req.num_frames)
        generator = torch.Generator("cpu").manual_seed(req.seed) if req.seed is not None else None

        print(f"[wan] t2v | {req.width}x{req.height} → {width}x{height} | frames={num_frames} steps={steps} cfg={guidance}", flush=True)

        result = _call_with_progress(
            pipe,
            prompt=req.prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=generator,
        )
        return result.frames[0]

    def image_to_video(self, pipes: tuple, req, ref_img: PILImage.Image) -> list:
        pipe, _ = pipes
        steps = req.num_inference_steps if req.num_inference_steps != 4 else self.default_steps
        guidance = req.guidance_scale if req.guidance_scale != 0.0 else self.default_guidance
        width, height = _wan_resolution(req.width, req.height)
        num_frames = _wan_frames(req.num_frames)
        generator = torch.Generator("cpu").manual_seed(req.seed) if req.seed is not None else None

        print(f"[wan] i2v | {req.width}x{req.height} → {width}x{height} | frames={num_frames} steps={steps} cfg={guidance}", flush=True)

        result = _call_with_progress(
            pipe,
            image=ref_img,
            prompt=req.prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=generator,
        )
        return result.frames[0]

    def capabilities(self) -> dict:
        return {
            "supports_i2i": self.supports_i2i,
            "is_video": self.is_video,
            "default_steps": self.default_steps,
            "default_guidance": self.default_guidance,
        }
