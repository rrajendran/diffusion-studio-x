import torch
from diffusers import DiffusionPipeline

MODEL_ID = "zai-org/GLM-Image"

ASPECT_RATIOS = {
    "1:1":  (512, 512),
    "4:3":  (768, 576),
    "16:9": (1024, 576),
    "3:4":  (576, 768),
    "9:16": (576, 1024),
}

def get_dimensions(ratio="16:9"):
    return ASPECT_RATIOS.get(ratio, ASPECT_RATIOS["1:1"])

def load_pipeline():
    pipe = DiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )

    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
    else:
        pipe = pipe.to("cpu")

    return pipe


def generate_image(
    prompt,
    width=1024,
    height=576,
    steps=30,
    guidance_scale=7.5,
    seed=None,
    output_file="output.png"
):
    pipe = load_pipeline()

    generator = None
    if seed is not None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        generator = torch.Generator(device).manual_seed(seed)

    image = pipe(
        prompt=prompt,
        width=width,
        height=height,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        generator=generator
    ).images[0]

    image.save(output_file)
    print(f"✅ Image saved to {output_file}")


if __name__ == "__main__":
    prompt = "A cinematic wide shot of a futuristic city at sunset, ultra detailed, 8k"
    width, height = get_dimensions("16:9")  # Returns (1024, 576)
    generate_image(
        prompt,
        width=width,
        height=height,
        steps=28,
        guidance_scale=7.5,
        seed=42
    )