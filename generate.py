import torch
import click

from diffusers import StableDiffusionPipeline
from torch import autocast


@click.command()
@click.option("--out", default="result.png", help="Name of the output file.")
@click.option("--seed", default=1234, help="Random seed to use.")
@click.option("--steps", default=50, help="Number of diffusion steps, more=better but takse longer.")
@click.option("--scale", default=8., help="Stable diffusion guidance scale.")
@click.argument("prompt")
def main(prompt: str, out:str, seed: int, steps: int, scale: float):
    # make sure you're logged in with `huggingface-cli login`
    pipe = StableDiffusionPipeline.from_pretrained("./stable-diffusion-v1-5")
    pipe = pipe.to("cuda")
    
    generator = torch.Generator("cuda").manual_seed(seed)
    with autocast("cuda"):
        image = pipe(prompt, num_inference_steps=steps, guidance_scale=scale, generator=generator).images[0]

    image.save(out)

if __name__ == "__main__":
    main()

