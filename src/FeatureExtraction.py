import cv2
import numpy as np
import csv


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
            img['Horizontal Symmetry'] = 'Yes'
        else:
            img['Horizontal Symmetry'] = 'No'


def verticalSymmetry(listOfCharacters):
    for img in listOfCharacters:
        image = img['image']
        processedImg = 'ProcessedImages' + image[image.index('/'):]
        processed = cv2.imread(processedImg)
        blackLeft = np.sum(processed[:, 0:15] == 0)
        blackRight = np.sum(processed[:, 15:30] == 0)
        ratio = blackLeft / blackRight
        if 0.8 <= ratio <= 1.2:
            img['Vertical Symmetry'] = 'Yes'
        else:
            img['Vertical Symmetry'] = 'No'


def main():
    chars = getListOfCharacters()
    getBlackToWhiteRatio(chars)
    horizontalSymmetry(chars)
    verticalSymmetry(chars)
    print(chars)


if __name__ == '__main__':
    main()
