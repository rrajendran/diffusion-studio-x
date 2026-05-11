"""Base class for all model adapters."""

import abc
from PIL import Image as PILImage


class ModelAdapter(abc.ABC):
    default_steps: int = 4
    default_guidance: float = 0.0
    supports_i2i: bool = False

    @classmethod
    @abc.abstractmethod
    def matches(cls, model_path: str) -> bool:
        """Return True if this adapter owns the given model path."""

    @abc.abstractmethod
    def load(self, model_path: str) -> tuple:
        """Load the model; return (pipe, i2i_pipe). i2i_pipe may be None."""

    @abc.abstractmethod
    def text_to_image(self, pipes: tuple, req) -> PILImage.Image:
        """Run text-to-image inference; return a PIL Image."""

    def image_to_image(self, pipes: tuple, req, ref_img: PILImage.Image) -> PILImage.Image:
        raise ValueError(f"Adapter {self.__class__.__name__} does not support image-to-image")

    def capabilities(self) -> dict:
        return {
            "supports_i2i": self.supports_i2i,
            "default_steps": self.default_steps,
            "default_guidance": self.default_guidance,
        }
