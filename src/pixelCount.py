import cv2
import numpy as np
import csv
import os


listOfDicts = []
excelFile = 'english.csv'
with open(excelFile, 'r') as data:
    for line in csv.DictReader(data):
        listOfDicts.append(line)


for img in listOfDicts:
    image = img['image']
    processedImg = 'ProcessedImages' + image[image.index('/'):]
    processed = cv2.imread(processedImg)
    blackToWhiteRatio = np.sum(processed == 0) / np.sum(processed == 255)
    img['blackToWhite'] = blackToWhiteRatio

print(listOfDicts)
