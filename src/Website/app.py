from flask import Flask, flash, request, redirect, render_template
import base64
from io import BytesIO
from PIL import Image
import joblib
from ImageToFeature import *
from src.GetBoundingRectange import *
from sequentialmodel import SequentialModel2

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'

app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])


svmModel = joblib.load('SVC.npy')
knnModel = joblib.load('KNN.npy')
forestModel = joblib.load('Forest.npy')


def find_top_3(predictions_proba):
    characters = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    predictions = []

    predictions_proba = [round(member, 2) for member in predictions_proba]

    max_number = max(predictions_proba)
    index1 = predictions_proba.index(max_number)
    predictions.append(characters[index1])
    predictions.append(predictions_proba[index1])
    predictions_proba[index1] = 0

    index2 = predictions_proba.index(max(predictions_proba))
    predictions.append(characters[index2])
    predictions.append(predictions_proba[index2])
    predictions_proba[index2] = 0

    index3 = predictions_proba.index(max(predictions_proba))
    predictions.append(characters[index3])
    predictions.append(predictions_proba[index3])
    predictions_proba[index3] = 0

    return predictions_proba, predictions


def Parallel_Ensemble(inputProfileWhite, inputHistWhite, model1, model2, model3, w1, w2, w3):
    y_model1_proba = model1.predict_proba(inputProfileWhite)
    y_model1_proba = y_model1_proba[0]
    y_model1_proba = [value * w1 for value in y_model1_proba]

    y_model2_proba = model2.predict_proba(inputHistWhite)
    y_model2_proba = y_model2_proba[0]
    y_model2_proba = [value * w2 for value in y_model2_proba]

    y_model3_proba = model3.predict_proba(inputProfileWhite)
    y_model3_proba = y_model3_proba[0]
    y_model3_proba = [value * w3 for value in y_model3_proba]

    final_y_proba = sum = [x + y for (x, y) in zip(y_model1_proba, y_model2_proba)]
    final_y_proba = sum = [x + y for (x, y) in zip(final_y_proba, y_model3_proba)]

    _, predictions = find_top_3(final_y_proba)

    return predictions


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
        userHistWhite, userProfileWhite = getFeatures('static/uploads/canvasImage.png')
        if model == 'svm-ensemble':
            probas = SequentialModel2(userProfileWhite)
        elif model == 'svm':
            y_new = svmModel.predict_proba(userProfileWhite)
            _, probas = find_top_3(y_new.tolist()[0])
        elif model == 'knn':
            y_new = knnModel.predict_proba(userHistWhite)
            _, probas = find_top_3(y_new.tolist()[0])
        elif model == 'random-forest':
            y_new = forestModel.predict_proba(userProfileWhite)
            _, probas = find_top_3(y_new.tolist()[0])
        elif model == 'ensemble':
            svmWeight = request.form['svm-weight']
            knnWeight = request.form['knn-weight']
            forestWeight = request.form['dt-weight']
            if svmWeight == '':
                svmWeight = 0
            if knnWeight == '':
                knnWeight = 0
            if forestWeight == '':
                forestWeight = 0
            svmWeight = float(svmWeight); knnWeight = float(knnWeight); forestWeight = float(forestWeight)
            if svmWeight + knnWeight + forestWeight != 1:
                flash('Total weight should equal 1', 'error')
                return redirect(request.url)
            probas = Parallel_Ensemble(userProfileWhite, userHistWhite, svmModel, knnModel, forestModel, svmWeight, knnWeight, forestWeight)
        _, _, boundingRectInput = processUserImage('static/uploads/canvasImage.png')
        cv2.imwrite('static/uploads/canvasBox.png', boundingRectInput)
        return render_template('index.html', outputLetter=probas, model=model, inputImage='static/uploads/canvasBox.png')
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
        userHistWhite, userProfileWhite = getFeatures('static/uploads/img.png')
        if model == 'svm-ensemble':
            probas = SequentialModel2(userProfileWhite)
        elif model == 'svm':
            y_new = svmModel.predict_proba(userProfileWhite)
            _, probas = find_top_3(y_new.tolist()[0])
        elif model == 'knn':
            y_new = knnModel.predict_proba(userHistWhite)
            _, probas = find_top_3(y_new.tolist()[0])
        elif model == 'random-forest':
            y_new = forestModel.predict_proba(userProfileWhite)
            _, probas = find_top_3(y_new.tolist()[0])
        elif model == 'ensemble':
            svmWeight = request.form['svm-weight']
            knnWeight = request.form['knn-weight']
            forestWeight = request.form['dt-weight']
            if svmWeight == '':
                svmWeight = 0
            if knnWeight == '':
                knnWeight = 0
            if forestWeight == '':
                forestWeight = 0
            svmWeight = float(svmWeight); knnWeight = float(knnWeight); forestWeight = float(forestWeight)
            if svmWeight + knnWeight + forestWeight != 1:
                flash('Total weight should equal 1', 'error')
                return redirect(request.url)
            probas = Parallel_Ensemble(userProfileWhite, userHistWhite, svmModel, knnModel, forestModel, svmWeight, knnWeight, forestWeight)
        _, _, boundingRectInput = processUserImage('static/uploads/img.png')
        cv2.imwrite('static/uploads/uploadBox.png', boundingRectInput)
        return render_template('index.html', outputLetter=probas, model=model, inputImage='static/uploads/uploadBox.png')
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
    app.run(debug=True, host="0.0.0.0")
