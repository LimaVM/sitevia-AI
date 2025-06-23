document.addEventListener('DOMContentLoaded', () => {
  const fileInput = document.getElementById('file-input');
  const preview = document.getElementById('preview');
  const form = document.getElementById('generate-form');
  const progress = document.getElementById('progress');

  if (fileInput) {
    fileInput.addEventListener('change', () => {
      const file = fileInput.files[0];
      if (file) {
        const reader = new FileReader();
        reader.onload = e => {
          preview.innerHTML = `<img src="${e.target.result}" width="256">`;
        };
        reader.readAsDataURL(file);
      }
    });
  }

  if (form && progress) {
    form.addEventListener('submit', () => {
      progress.style.display = 'block';
    });
  }
});
