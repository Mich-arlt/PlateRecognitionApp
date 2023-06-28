import os
from flask import Flask, render_template, request

import deeplearning

app = Flask(__name__)

BASE_PATH = os.getcwd()
UPLOAD_PATH = os.path.join(BASE_PATH, 'static/upload/')

@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        upload_files = request.files.getlist('image_name')
        image_paths = []
        texts = []
        for upload_file in upload_files:
            filename = upload_file.filename
            path_save = os.path.join(UPLOAD_PATH, filename)
            upload_file.save(path_save)
            text = deeplearning.OCR(path_save, filename)
            image_paths.append(filename)
            texts.append(text)

        return render_template('index.html', upload=True, upload_images=image_paths, texts=texts)
    return render_template('index.html', upload=False)

if __name__=="__main__":
    app.run(debug=True)