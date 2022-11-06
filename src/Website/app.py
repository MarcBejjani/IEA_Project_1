from flask import Flask, flash, request, redirect, render_template
import base64
from io import BytesIO
from PIL import Image
import joblib
from ImageToFeature import *
from src.GetBoundingRectange import *
from sequentialmodel import SequentialModel

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'

app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])


svmModel = joblib.load('savedSVM.sav')


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/', methods=['GET', 'POST'])
def upload_image():
    inputType = request.form['inputType']
    if inputType == 'drawing':
        model = request.form['model']
        userInput = getFeatures('static/uploads/canvasImage.png')
        if model == 'svm-ensemble':
            y_new = SequentialModel(userInput)
        elif model == 'svm':
            y_new = svmModel.predict(userInput)
        elif model == 'knn':
            pass
        elif model == 'random-forest':
            pass
        elif model == 'ensemble':
            svmWeight = request.form['svm-weight']
            knnWeight = request.form['knn-weight']
            dtWeight = request.form['dt-weight']
        _, _, boundingRectInput = processUserImage('static/uploads/canvasImage.png')
        cv2.imwrite('static/uploads/canvasBox.png', boundingRectInput)
        return render_template('index.html', outputLetter=y_new, model=model, inputImage='static/uploads/canvasBox.png')
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = 'img.png'
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        model = request.form['model']
        userInput = getFeatures('static/uploads/img.png')
        if model == 'svm-ensemble':
            y_new = SequentialModel(userInput)
        elif model == 'svm':
            y_new = svmModel.predict(userInput)
        elif model == 'knn':
            pass
        elif model == 'random-forest':
            pass
        elif model == 'ensemble':
            svmWeight = request.form['svm-weight']
            knnWeight = request.form['knn-weight']
            dtWeight = request.form['dt-weight']
        _, _, boundingRectInput = processUserImage('static/uploads/img.png')
        cv2.imwrite('static/uploads/uploadBox.png', boundingRectInput)
        return render_template('index.html', outputLetter=y_new, model=model, inputImage='static/uploads/uploadBox.png')
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(request.url)


@app.route('/draw')
def display_image():
    return render_template('drawing.html')


@app.route('/draw', methods=['GET', 'POST'])
def draw():
    canvasImage = request.form['js_data']
    offset = canvasImage.index(',') + 1
    img_bytes = base64.b64decode(canvasImage[offset:])
    img = Image.open(BytesIO(img_bytes))
    img = np.array(img)
    cv2.imwrite('static/uploads/canvasImage.png', img)
    return render_template('drawing.html')


if __name__ == "__main__":
    app.run()
