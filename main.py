import os

import numpy as np
from flask import flash, request, redirect, render_template

from setuptools import glob
from tensorflow_core.python.keras.models import model_from_json
from tensorflow_core.python.keras.preprocessing import image
from werkzeug.utils import secure_filename
from app import app

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def upload_form():
    return render_template('index2.html')


@app.route('/', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No file selected for uploading')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            flash('File successfully uploaded')
            return redirect('/')
        else:
            flash('Allowed file types are png, jpg, jpeg')
            return redirect(request.url)

IMAGE_SIZE = 50

@app.route('/predict',methods=['GET','POST'])
def predict():
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")
    path = "C://Users/Mrudula Bapat/PycharmProjects/cnnmodelv1/uploads/*.*"
    images = []
    filename = []
    for file in glob.glob(path):
        # print(file)
        filename.append(file)
        img = image.load_img(file, target_size=(50, 50))
        img_tensor = image.img_to_array(img)
        img_tensor = np.expand_dims(img_tensor, axis=0)
        img_tensor /= 255
        images.append(img_tensor)
    # print(images)
    predImg = np.vstack(images)
    predictions = loaded_model.predict(predImg)
    return filename, predictions


@app.route('/predict_api',methods=['GET','POST'])
def predict_api():
    pathupload = "C://Users/Mrudula Bapat/PycharmProjects/cnnmodelv1/uploads"
    dir = os.listdir(pathupload)
    if request.method == 'POST' and len(dir) != 0:
        filename, predictions = predict()
        return render_template('index2.html', filename=filename,predictions=predictions, len=len(filename))
    elif len(dir) == 0:
        flash('No file selected for uploading')
        return redirect('/')
    else:
        return

@app.route('/knowmore', methods = ['POST'])
def knowmore():
    return render_template("knowmore.html")


if __name__ == "__main__":
    app.run(debug=True)