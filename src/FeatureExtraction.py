import cv2
import numpy as np
import csv
import pandas as pd
from matplotlib import pyplot as plt


def getListOfCharacters():
    listOfCharacters = []
    excelFile = 'english.csv'
    with open(excelFile, 'r') as data:
        for line in csv.DictReader(data):
            listOfCharacters.append(line)
    return listOfCharacters


def getBlackToWhiteRatio(listOfCharacters):
    for img in listOfCharacters:
        image = img['image']
        processedImg = 'ProcessedImages' + image[image.index('/'):]
        processed = cv2.imread(processedImg)
        blackToWhiteRatio = np.sum(processed == 0) / np.sum(processed == 255)
        img['blackToWhite'] = blackToWhiteRatio


def horizontalSymmetry(listOfCharacters):
    for img in listOfCharacters:
        image = img['image']
        processedImg = 'ProcessedImages' + image[image.index('/'):]
        processed = cv2.imread(processedImg)
        blackTop = np.sum(processed[0:15, :] == 0)
        blackBottom = np.sum(processed[15:30, :] == 0)
        ratio = blackTop / blackBottom
        if 0.8 <= ratio <= 1.2:
            img['Horizontal Symmetry'] = '1'
        else:
            img['Horizontal Symmetry'] = '0'


def inverseSymmetry(listOfCharacters):
    for img in listOfCharacters:
        image = img['image']
        processedImg = 'ProcessedImages' + image[image.index('/'):]
        processed = cv2.imread(processedImg)
        blackTopLeft = np.sum(processed[0:15, 0:15] == 0)
        blackTopRight = np.sum(processed[0:15, 15:30] == 0)
        blackBottomLeft = np.sum(processed[15:30, 0:15] == 0) + 0.000000000000001
        blackBottomRight = np.sum(processed[15:30, 15:30] == 0) + 0.000000000000001
        ratio1 = blackTopLeft / blackBottomRight
        ratio2 = blackTopRight / blackBottomLeft
        img['Inverse Symmetry'] = '0'
        if 0.8 <= ratio1 and ratio2 <= 1.2:
            img['Inverse Symmetry'] = '1'


def verticalSymmetry(listOfCharacters):
    for img in listOfCharacters:
        image = img['image']
        processedImg = 'ProcessedImages' + image[image.index('/'):]
        processed = cv2.imread(processedImg)
        blackLeft = np.sum(processed[:, 0:15] == 0)
        blackRight = np.sum(processed[:, 15:30] == 0)
        ratio = blackLeft / blackRight
        if 0.9 <= ratio <= 1.1:
            img['Vertical Symmetry'] = '1'
        else:
            img['Vertical Symmetry'] = '0'


def getAspectRatio(listOfCharacters):
    for img in listOfCharacters:
        image = img['image']
        processedImg = 'ProcessedImages' + image[image.index('/'):]
        processed = cv2.imread(processedImg)
        x1, y1, w, h = cv2.boundingRect(processed)
        aspectRatio = w / h
        img['Aspect Ratio'] = aspectRatio


def projectionHistogram (image):
    y_axis = np.sum(image, axis=0) #sum the values in each column of the image

    dimensions = image.shape
    print(dimensions)
    x_axis = np.arange(0, dimensions[1])

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert BGR to RBG since open cv is BGR and matplot is RGB
    plt.subplot(2, 1, 1)
    plt.imshow(image)
    #cv2.imshow("final image", image)

    plt.subplot(2,1,2)
    plt.bar(x_axis, y_axis, color = 'red', width= 0.4)
    plt.show()

    return x_axis, y_axis


def profile (image):
    dimensions = image.shape  # (m,n) m:rows/ n:columns
    #print(dimensions)
    """
            #####   Bottom  #####
    """
    bottom = []
    bottom_sum = 0

    for j in range(dimensions[1]): #columns
        for i in reversed(range(dimensions[0])): #rows
            if image[i][j] == 0:
                bottom_sum = bottom_sum + 1
            elif image[i][j] ==255:
                break
        bottom = np.append(bottom, bottom_sum)
        bottom_sum = 0

    x_axis = np.arange(0,dimensions[1])

    plt.subplot(1, 4, 1)
    plt.bar(x_axis, bottom, color='red', width=0.5)
    plt.title("bottom profile")

    """
            #####   TOP  #####
    """
    top = []
    top_sum = 0

    for j in range(dimensions[1]): #columns
        for i in range(dimensions[0]): #rows
            if image[i][j] == 0:
                top_sum = top_sum + 1
            elif image[i][j] == 255:
                break
        # print(f"bottom_sum {bottom_sum}")
        top = np.append(top, top_sum)
        # print(f"bottom {bottom}")
        top_sum = 0

    plt.subplot(1, 4, 2)
    plt.bar(x_axis, top, color='red', width=0.5)
    plt.title("top profile")

    """
            #####  RIGHT  #####
    """
    right = []
    right_sum = 0

    for i in range(dimensions[0]): # rows
        for j in reversed(range(dimensions[1])): #columns
            if image[i][j] == 0:
                right_sum = right_sum + 1
            elif image[i][j] == 255:
                break
        right = np.append(right, right_sum)
        right_sum = 0

    y_axis = np.arange(0, dimensions[0])

    plt.subplot(1, 4, 3)
    plt.bar(y_axis, right, color='red', width=0.5)
    plt.title("right profile")

    """
            #####  left #####
    """
    left = []
    left_sum = 0

    for i in range(dimensions[0]):  # rows
        for j in range(dimensions[1]):  # columns
            if image[i][j] == 0:
                left_sum = left_sum + 1
            elif image[i][j] == 255:
                break
        left = np.append(left, left_sum)
        left_sum = 0

    y_axis = np.arange(0, dimensions[0])

    return x_axis, y_axis, bottom, top, right, left


def main():
    chars = getListOfCharacters()
    getBlackToWhiteRatio(chars)
    horizontalSymmetry(chars)
    verticalSymmetry(chars)
    inverseSymmetry(chars)
    #getAspectRatio(chars)
    print(chars)
    return chars


if __name__ == '__main__':
    set = main()
    df = pd.DataFrame(set)
    df.to_csv('dataSet.csv', index=False, header=True)
