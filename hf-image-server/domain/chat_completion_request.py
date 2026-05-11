"""
Domain models for the image generation server.
"""

from pydantic import BaseModel

from .chat_message import ChatMessage


class ChatCompletionRequest(BaseModel):
    model: str = ""
    messages: list[ChatMessage]
    max_tokens: int = 512
    temperature: float = 0.7
    stream: bool = False