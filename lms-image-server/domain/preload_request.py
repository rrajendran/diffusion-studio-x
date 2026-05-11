from pydantic import BaseModel


class PreloadRequest(BaseModel):
    model: str   # full absolute path to the GGUF/MLX model
