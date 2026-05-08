"""
Base model adapter for the image generation server.
"""

import abc
from PIL import Image as PILImage


class ModelAdapter(abc.ABC):
    """
    Contract for a model family adapter.

    Subclasses must implement:
      matches()        — decides ownership of a model_id
      load()           — returns (t2i_pipe, i2i_pipe); i2i_pipe may be None
      text_to_image()  — runs inference and returns a PIL Image

    Subclasses may implement:
      image_to_image() — required when supports_i2i = True

    Class-level attributes set sensible per-family defaults for steps/guidance.
    """

    default_steps: int = 4
    default_guidance: float = 0.0
    supports_i2i: bool = False

    @classmethod
    @abc.abstractmethod
    def matches(cls, model_id: str) -> bool:
        """Return True if this adapter handles the given model_id."""

    @abc.abstractmethod
    def load(self, model_id: str) -> tuple:
        """
        Load and return (t2i_pipe, i2i_pipe).
        i2i_pipe may be None for text-only models.
        Call _apply_pipeline_optimizations() before returning.
        """

    @abc.abstractmethod
    def text_to_image(self, pipes: tuple, req) -> PILImage.Image:
        """Run text-to-image inference; return a PIL Image."""

    def image_to_image(self, pipes: tuple, req, ref_img: PILImage.Image) -> PILImage.Image:
        raise ValueError(f"Model family does not support image-to-image: {req.model}")

    def capabilities(self) -> dict:
        return {
            "supports_i2i": self.supports_i2i,
            "default_steps": self.default_steps,
            "default_guidance": self.default_guidance,
        }