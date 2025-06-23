# GPU Web Test

This simple Flask application allows you to upload an image and generate a new image or short video using Stable Diffusion models. Enter a text prompt and choose whether to create an image or animate the picture as a video.

## Requirements

- Python 3.8+
- `flask`, `torch`, `torchvision`, `pillow`
- `diffusers`, `transformers`, `accelerate` *(for generation)*

Install dependencies with:

```bash
pip install flask torch torchvision pillow
```

For image/video generation install:

```bash
pip install diffusers transformers accelerate
```

## Running

```bash
python app.py
```

Then open `http://localhost:45600` in your browser. Upload an image, enter a text
prompt and choose whether you want to generate a new image or a short video.

## Generating images and videos

Use `generate.py` with a text prompt and an initial image:

```bash
python generate.py "A beautiful sunset" --init input.png --output sunset.png
```

Generate a short video (requires more VRAM):

```bash
python generate.py "A cat walking" --init input.png --video --output cat.mp4 --frames 16
```

The pipelines use float16 weights and attention slicing which run on GPUs with around 8GB of VRAM.
