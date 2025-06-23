from flask import Flask, render_template, request, send_from_directory
import torch
from PIL import Image
import torchvision.transforms as transforms
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Exemplo simples de transformação (converte para escala de cinza usando GPU)
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor()
])

@app.route('/', methods=['GET', 'POST'])
def index():
    result_image = None
    if request.method == 'POST':
        if 'image' in request.files:
            file = request.files['image']
            if file.filename != '':
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(filepath)

                img = Image.open(filepath).convert('RGB')
                img_tensor = transform(img).unsqueeze(0).to('cuda')

                # Exemplo simples: só manda a imagem de volta, já redimensionada e cinza
                output_img = img_tensor.squeeze().cpu().permute(1, 2, 0).numpy()
                output_img = (output_img * 255).astype('uint8')
                output = Image.fromarray(output_img)

                output_path = os.path.join(RESULT_FOLDER, 'result.jpg')
                output.save(output_path)
                result_image = output_path

    return render_template('index.html', result_image=result_image)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=45600, debug=True)
