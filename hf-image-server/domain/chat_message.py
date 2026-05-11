"""
Domain models for the image generation server.
"""

from pydantic import BaseModel


class ChatMessage(BaseModel):
    role: str
    content: str