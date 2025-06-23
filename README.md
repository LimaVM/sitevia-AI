# GPU Web Test

This simple Flask application allows you to upload an image, processes it using PyTorch and shows the result in grayscale. The original and processed images are displayed side by side and you can download the processed result.

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

Then open `http://localhost:45600` in your browser.

## Generating images and videos

Use `generate.py` with a text prompt:

```bash
python generate.py "A beautiful sunset" --output sunset.png
```

Generate a short video (requires more VRAM):

```bash
python generate.py "A cat walking" --video --output cat.mp4 --frames 16
```

The pipelines use float16 weights and attention slicing which run on GPUs with around 8GB of VRAM.
