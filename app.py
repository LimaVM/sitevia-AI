from flask import Flask, render_template, request
import torch
from PIL import Image
import torchvision.transforms as transforms
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor()
])

@app.route('/', methods=['GET', 'POST'])
def index():
    result_image = None
    original_image = None
    if request.method == 'POST':
        if 'image' in request.files:
            file = request.files['image']
            if file.filename != '':
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(filepath)

                img = Image.open(filepath).convert('RGB')
                img_tensor = transform(img).unsqueeze(0).to(device)

                output_img = img_tensor.squeeze().cpu().permute(1, 2, 0).numpy()
                output_img = (output_img * 255).astype('uint8')
                output = Image.fromarray(output_img)

                output_path = os.path.join(RESULT_FOLDER, 'result.jpg')
                output.save(output_path)
                result_image = output_path
                original_image = filepath

    return render_template('index.html', result_image=result_image, original_image=original_image)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=45600, debug=True)
