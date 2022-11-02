import pandas as pd
from src.FeatureExtraction import *


def processImage(dir):
    image = cv2.imread(dir)
    image = BGR2BINARY(image)
    boundingBox = getBoundingRect(image)
    boxWithWhite = resizeToSquare(boundingBox)
    boxNoWhite = resizeImage(boundingBox, 30, 30)
    return boundingBox, boxNoWhite, boxWithWhite


def getFeatures(dir):

    boundingBox, boxNoWhite, boxWithWhite = processImage(dir)

    inputDict = {'BlackToWhite': getBlackToWhiteRatio(boundingBox),
                 'Aspect Ratio': getAspectRatio(boundingBox)}

    column_sum, row_sum = getProjectionHistogram(boxNoWhite)
    for idx1, value in enumerate(column_sum):
        inputDict[f'Column Histogram {idx1}'] = value

    for idx1, value in enumerate(row_sum):
        inputDict[f'Row Histogram {idx1}'] = value

    # Profile
    bottom, top, left, right = getProfile(boxNoWhite)
    for idx1, value in enumerate(bottom):
        inputDict[f'Bottom Profile {idx1}'] = value

    for idx1, value in enumerate(top):
        inputDict[f'Top Profile {idx1}'] = value

    for idx1, value in enumerate(left):
        inputDict[f'Left Profile {idx1}'] = value

    for idx1, value in enumerate(right):
        inputDict[f'Right Profile {idx1}'] = value

    # Hog
    image_hog = getHOG(boxNoWhite)
    for idx1, value in enumerate(image_hog):
        inputDict[f'HOG {idx1}'] = value

    inputDict = pd.DataFrame(inputDict, index=[0])

    return inputDict



if __name__ == '__main__':
    userInput = getFeatures('static/uploads/img.png')
    print(userInput)
