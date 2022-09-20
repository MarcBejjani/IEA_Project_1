import cv2
import numpy as np
import os

dirname, filename = os.path.split(os.path.abspath(__file__))

frame = cv2.imread(dirname+'\EnglishHandwrittenCharacters\img001-001.png',-1)
cv2.imshow('Normal frame', frame)

def BGR2BINARY (image, threshold):
    gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    thresh_image = cv2.threshold(gray_image,155,256,cv2.THRESH_BINARY)[1]
    binary_image = cv2.bitwise_not(thresh_image)
    return binary_image

binary_image = BGR2BINARY(frame, 0)
cv2.imshow('binary frame', binary_image)

def getBoundingRect(image):
    x1,y1,w,h = cv2.boundingRect(image)
    x2 = x1+w
    y2 = y1+h

    bounding_rect_image = image [y1:y2,x1:x2]
    bounding_rect_image = cv2.bitwise_not(bounding_rect_image)
    return bounding_rect_image

boundingBox = getBoundingRect(binary_image)
cv2.imshow('bounding frame', boundingBox)

cv2.waitKey(0)
cv2.destroyAllWindows()