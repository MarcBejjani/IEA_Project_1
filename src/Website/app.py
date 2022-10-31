from flask import Flask, flash, request, redirect, url_for, render_template
import urllib.request
import os
from werkzeug.utils import secure_filename
import sys
sys.path.append('../src')
from src.GetBoundingRectange import *

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'

app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        # filename = secure_filename(file.filename)
        filename = 'img.png'
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        # print('upload_image filename: ' + filename)
        flash('Image successfully uploaded and displayed below')
        processedImage = cv2.imread('static/uploads/img.png')
        processedImage = BGR2BINARY(processedImage)
        procName = 'processed.png'
        cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], procName), processedImage)
        return render_template('index.html', filename=procName)
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(request.url)


@app.route('/display/<filename>')
def display_image(filename):
    # print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)


@app.route('/imageprocessing')
def imageprocessing():
    return render_template('process.html', inputPath='static/uploads/img.png')


@app.route('/imageprocessing', methods=['GET', 'POST'])
def process():
    if request.method == 'POST':
        if request.form.get('action1') == 'VALUE1':
            inputPath = 'static/uploads/img.png'
            inputImage = cv2.imread(inputPath)
            procImage = BGR2BINARY(inputImage)
            name = 'processed.png'
            cv2.imwrite(f'static/uploads/{name}', procImage)
        return render_template("process.html", procImage=procImage)

    return redirect(request.url)


if __name__ == "__main__":
    app.run()
