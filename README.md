# GPU Web Test

This simple Flask application allows you to upload an image, processes it using PyTorch and shows the result in grayscale. The original and processed images are displayed side by side and you can download the processed result.

## Requirements

- Python 3.8+
- `flask`, `torch`, `torchvision`, `pillow`

Install dependencies with:

```bash
pip install flask torch torchvision pillow
```

## Running

```bash
python app.py
```

Then open `http://localhost:45600` in your browser.
