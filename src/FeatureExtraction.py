import cv2
import numpy as np
import csv
import pandas as pd
from skimage.feature import hog
from GetBoundingRectange import BGR2BINARY


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
        processedImg = 'BoundingRectangleImages' + image[image.index('/'):]
        processed = cv2.imread(processedImg)
        processed = BGR2BINARY(processed)
        blackToWhiteRatio = np.sum(processed == 0) / np.sum(processed == 255)
        img['blackToWhite'] = blackToWhiteRatio


def horizontalSymmetry(listOfCharacters):
    for img in listOfCharacters:
        image = img['image']
        processedImg = 'BoundingRectangleImages' + image[image.index('/'):]
        processed = cv2.imread(processedImg)
        processed = BGR2BINARY(processed)
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
        processedImg = 'BoundingRectangleImages' + image[image.index('/'):]
        processed = cv2.imread(processedImg)
        processed = BGR2BINARY(processed)
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
        processedImg = 'BoundingRectangleImages' + image[image.index('/'):]
        processed = cv2.imread(processedImg)
        processed = BGR2BINARY(processed)
        blackLeft = np.sum(processed[:, 0:15] == 0)
        blackRight = np.sum(processed[:, 15:30] == 0) + 0.0000000000000000001
        ratio = blackLeft / blackRight
        if 0.9 <= ratio <= 1.1:
            img['Vertical Symmetry'] = '1'
        else:
            img['Vertical Symmetry'] = '0'


def getAspectRatio(listOfCharacters):
    for img in listOfCharacters:
        image = img['image']
        processedImg = 'BoundingRectangleImages' + image[image.index('/'):]
        processed = cv2.imread(processedImg)
        processed = BGR2BINARY(processed)

        x1, y1, w, h = cv2.boundingRect(processed)
        aspectRatio = w / h
        img['Aspect Ratio'] = aspectRatio


def normalize_2d(matrix):
    norm = np.linalg.norm(matrix)
    normalized_matrix = matrix/norm

    return normalized_matrix


def projectionHistogram(listOfCharacters):
    for img in listOfCharacters:
        src = img['image']
        image = 'BoundingRectangleImages' + src[src.index('/'):]
        image = cv2.imread(image)

        image = cv2.bitwise_not(image)
        image = BGR2BINARY(image)
        #dimensions = image.shape

        #x_axis = np.arange(0, dimensions[1])
        column_sum = np.sum(image, axis=0)  # sum the values in each column of the img
        row_sum = np.sum(image, axis=1)  # sum the values in each row of the img

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert BGR to RBG since open cv is BGR and matplot is RGB

        column_sum = normalize_2d(column_sum)
        row_sum = normalize_2d(row_sum)

        flattenedColumnSum = column_sum.flatten()
        flattenedRowSum = row_sum.flatten()

        counter = 1
        for value in flattenedColumnSum:
            img[f'Column Histogram {counter}'] = value
            counter += 1
        counter = 1
        for value in flattenedRowSum:
            img[f'Row Histogram {counter}'] = value
            counter += 1


def profile(listOfCharacters):
    for img in listOfCharacters:
        imge = img['image']
        image = 'BoundingRectangleImages' + imge[imge.index('/'):]
        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        dimensions = image.shape  # (m,n) m:rows/ n:columns
        image = cv2.bitwise_not(image)

        """
                #####   Bottom  #####
        """
        bottom = []
        bottom_sum = 0

        for j in range(dimensions[1]):  # columns
            for i in reversed(range(dimensions[0])):  # rows
                if image[i][j] == 0:
                    bottom_sum = bottom_sum + 1
                elif image[i][j] == 255:
                    break
            bottom = np.append(bottom, bottom_sum)
            bottom_sum = 0

        x_axis = np.arange(0, dimensions[
            1])  # creating a matrix of values ranging from 0 till dimension[1] (width of image)
        bottom = normalize_2d(bottom)

        """
                #####   TOP  #####
        """
        top = []
        top_sum = 0

        for j in range(dimensions[1]):  # columns
            for i in range(dimensions[0]):  # rows
                if image[i][j] == 0:
                    top_sum = top_sum + 1
                elif image[i][j] == 255:
                    break
            top = np.append(top, top_sum)
            top_sum = 0
        top = normalize_2d(top)

        """
                #####  RIGHT  #####
        """
        right = []
        right_sum = 0

        for i in range(dimensions[0]):  # rows
            for j in reversed(range(dimensions[1])):  # columns
                if image[i][j] == 0:
                    right_sum = right_sum + 1
                elif image[i][j] == 255:
                    break
            right = np.append(right, right_sum)
            right_sum = 0

        right = normalize_2d(right)

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

        left = normalize_2d(left)

        y_axis = np.arange(0, dimensions[0])
        # profilePlot(x_axis, y_axis, bottom, top, left, right)

        counter = 1
        for white in bottom:
            img[f'Bottom {counter}'] = white
            counter += 1
        counter = 1
        for white in top:
            img[f'Top {counter}'] = white
            counter += 1
        counter = 1
        for white in left:
            img[f'Left {counter}'] = white
            counter += 1
        counter = 1
        for white in right:
            img[f'Right {counter}'] = white
            counter += 1


def HOG(listOfCharacters):
    for img in listOfCharacters:
        image = img['image']
        processedImg = 'BoundingRectangleImages' + image[image.index('/'):]
        processed = cv2.imread(processedImg)
        processed = BGR2BINARY(processed)
        resized_img = cv2.resize(processed, (128, 64))
        hog_feature, image_hog = hog(resized_img, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(4, 4), visualize=True)  # Set visualize to true if we need to see the image
        counter = 1
        for value in hog_feature:
            img[f'HOG {counter}'] = value
            counter += 1


def main():
    chars = getListOfCharacters()
    getBlackToWhiteRatio(chars)
    horizontalSymmetry(chars)
    verticalSymmetry(chars)
    inverseSymmetry(chars)
    profile(chars)
    projectionHistogram(chars)
    HOG(chars)
    getAspectRatio(chars)
    return chars


if __name__ == '__main__':
    set = main()
    df = pd.DataFrame(set)
    df.to_csv('dataSet.csv', index=False, header=True)