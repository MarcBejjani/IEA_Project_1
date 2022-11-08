import pandas as pd
from src.FeatureExtraction import *


def processUserImage(dir):
    image = cv2.imread(dir)
    image = BGR2BINARY(image, 3, 3)
    boundingBox = getBoundingRect(image)
    boxWithWhite = resizeToSquare(boundingBox)
    boxWithWhite = resizeImage(boxWithWhite, 30, 30)
    boxNoWhite = resizeImage(boundingBox, 30, 30)
    return boundingBox, boxNoWhite, boxWithWhite


def getFeatures(dir):

    boundingBox, boxNoWhite, boxWhite = processUserImage(dir)

    inputHistWhite = {'BlackToWhite': getBlackToWhiteRatio(boundingBox),
                 'Aspect Ratio': getAspectRatio(boundingBox)}

    inputProfileWhite = {'BlackToWhite': getBlackToWhiteRatio(boundingBox),
                      'Aspect Ratio': getAspectRatio(boundingBox)}

    column_sum, row_sum = getProjectionHistogram(boxWhite)
    for idx1, value in enumerate(column_sum):
        inputHistWhite[f'Column Histogram {idx1}'] = value

    for idx1, value in enumerate(row_sum):
        inputHistWhite[f'Row Histogram {idx1}'] = value

    column_sum, row_sum = getProjectionHistogram(boxNoWhite)
    for idx1, value in enumerate(column_sum):
        inputProfileWhite[f'Column Histogram {idx1}'] = value

    for idx1, value in enumerate(row_sum):
        inputProfileWhite[f'Row Histogram {idx1}'] = value

    # Profile
    bottom, top, left, right = getProfile(boxNoWhite)
    for idx1, value in enumerate(bottom):
        inputHistWhite[f'Bottom Profile {idx1}'] = value

    for idx1, value in enumerate(top):
        inputHistWhite[f'Top Profile {idx1}'] = value

    for idx1, value in enumerate(left):
        inputHistWhite[f'Left Profile {idx1}'] = value

    for idx1, value in enumerate(right):
        inputHistWhite[f'Right Profile {idx1}'] = value

    # Profile
    bottom, top, left, right = getProfile(boxWhite)
    for idx1, value in enumerate(bottom):
        inputProfileWhite[f'Bottom Profile {idx1}'] = value

    for idx1, value in enumerate(top):
        inputProfileWhite[f'Top Profile {idx1}'] = value

    for idx1, value in enumerate(left):
        inputProfileWhite[f'Left Profile {idx1}'] = value

    for idx1, value in enumerate(right):
        inputProfileWhite[f'Right Profile {idx1}'] = value

    # Hog
    image_hog = getHOG(boundingBox)
    for idx1, value in enumerate(image_hog):
        inputHistWhite[f'HOG {idx1}'] = value
        inputProfileWhite[f'HOG {idx1}'] = value

    inputHistWhite = pd.DataFrame(inputHistWhite, index=[0])
    inputProfileWhite = pd.DataFrame(inputProfileWhite, index=[0])

    return inputHistWhite, inputProfileWhite



if __name__ == '__main__':
    userInput = getFeatures('static/uploads/img.png')
    print(userInput)
