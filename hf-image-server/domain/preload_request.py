"""
Domain models for the image generation server.
"""

from pydantic import BaseModel


class PreloadRequest(BaseModel):
    model: str