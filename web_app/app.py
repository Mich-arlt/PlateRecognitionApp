import os
from flask import Flask, render_template, request

import deeplearning

app = Flask(__name__)

BASE_PATH = os.getcwd()
UPLOAD_PATH = os.path.join(BASE_PATH, 'static/upload/')
PREDICT_PATH = os.path.join(BASE_PATH, 'static/predict/')
ROI_PATH = os.path.join(BASE_PATH, 'static/roi/')

@app.route('/', methods=['POST', 'GET'])
def index():

    # stworz foldery pomocnicze jesli ich nie ma, lub wyczysc ich zawartosc, jesli sa
    for path in [UPLOAD_PATH, PREDICT_PATH, ROI_PATH]:
        if not os.path.exists(path):
            os.makedirs(path)
        else:
            folder_content = os.listdir(path)
            for item in folder_content:
                item_path = os.path.join(path, item)
                if os.path.isfile(item_path):
                    os.remove(item_path)

    if request.method == 'POST':
        upload_files = request.files.getlist('image_name')
        image_paths = []
        texts = []
        for upload_file in upload_files:
            filename = upload_file.filename
            path_save = os.path.join(UPLOAD_PATH, filename)
            upload_file.save(path_save)
            try:
                text = deeplearning.OCR(path_save, filename)
            except:
                text = ''
            image_paths.append(filename)
            texts.append(text)

        return render_template('index.html', upload=True, upload_images=image_paths, texts=texts)
    return render_template('index.html', upload=False)

if __name__=="__main__":
    app.run(debug=True)