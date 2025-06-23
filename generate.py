import argparse
import torch
from diffusers import StableDiffusionPipeline, StableVideoDiffusionPipeline


def generate_image(prompt: str, output: str, steps: int = 25):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
    )
    pipe = pipe.to(device)
    pipe.enable_attention_slicing()
    image = pipe(prompt, num_inference_steps=steps).images[0]
    image.save(output)
    print(f"Image saved to {output}")


def generate_video(prompt: str, output: str, frames: int = 16, steps: int = 25):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pipe = StableVideoDiffusionPipeline.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid",
        torch_dtype=torch.float16,
    )
    pipe = pipe.to(device)
    pipe.enable_attention_slicing()
    result = pipe(prompt, num_frames=frames, num_inference_steps=steps)
    result.frames[0].save(output)
    print(f"Video saved to {output}")


def main():
    parser = argparse.ArgumentParser(description="Generate images or video using Stable Diffusion")
    parser.add_argument("prompt", help="Text prompt")
    parser.add_argument("--video", action="store_true", help="Generate video instead of image")
    parser.add_argument("--output", default="output.png", help="Output file path")
    parser.add_argument("--steps", type=int, default=25, help="Inference steps")
    parser.add_argument("--frames", type=int, default=16, help="Number of frames for video")
    args = parser.parse_args()

    if args.video:
        generate_video(args.prompt, args.output, args.frames, args.steps)
    else:
        generate_image(args.prompt, args.output, args.steps)


if __name__ == "__main__":
    main()
