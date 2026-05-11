"""
Domain models for the image generation server.
"""

from pydantic import BaseModel, Field
import random


class GenerateRequest(BaseModel):
    prompt: str
    model: str = ""
    width: int = 1024
    height: int = 768
    num_inference_steps: int = 4    # sentinel — adapters substitute their own default
    guidance_scale: float = 0.0     # sentinel — adapters substitute their own default
    seed: int = Field(default_factory=lambda: random.randint(0, 2**32 - 1))
    reference_image: str | None = None  # base64 data URL; triggers i2i/i2v when present
    num_frames: int = 49            # for video models; sentinel value
    fps: int = 16                   # output video frame rate