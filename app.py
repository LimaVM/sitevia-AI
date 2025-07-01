from flask import Flask, render_template, request
import torch
from PIL import Image
import torchvision.transforms as transforms
from diffusers import StableDiffusionImg2ImgPipeline, StableVideoDiffusionPipeline
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# lazy loaded pipelines
img_pipe = None
vid_pipe = None


transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
])

def get_img_pipe():
    global img_pipe
    if img_pipe is None:
        img_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16,
        )
        img_pipe = img_pipe.to(device)
        img_pipe.enable_attention_slicing()
    return img_pipe


def get_vid_pipe():
    global vid_pipe
    if vid_pipe is None:
        vid_pipe = StableVideoDiffusionPipeline.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid",
            torch_dtype=torch.float16,
        )
        vid_pipe = vid_pipe.to(device)
        vid_pipe.enable_attention_slicing()
    return vid_pipe

@app.route('/', methods=['GET', 'POST'])
def index():
    result_image = None
    result_video = None
    original_image = None
    if request.method == 'POST':
        mode = request.form.get('mode', 'image')
        prompt = request.form.get('prompt', '')
        file = request.files.get('image')
        if file and allowed_file(file.filename):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            img = Image.open(filepath).convert('RGB')
            img = img.resize((512, 512))
            if mode == 'video':
                pipe = get_vid_pipe()
                result = pipe(image=img, num_frames=16, num_inference_steps=25)
                output_path = os.path.join(RESULT_FOLDER, 'result.mp4')
                result.frames[0].save(output_path)
                result_video = output_path
            else:
                pipe = get_img_pipe()
                result = pipe(prompt=prompt, image=img, strength=0.8, num_inference_steps=25)
                output_path = os.path.join(RESULT_FOLDER, 'result.png')
                result.images[0].save(output_path)
                result_image = output_path

            original_image = filepath

    return render_template('index.html', result_image=result_image, result_video=result_video, original_image=original_image)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=45600, debug=True)
