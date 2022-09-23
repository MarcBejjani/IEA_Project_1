import cv2
import numpy as np
import os

"""
A function to get the binary black and white image from a RGB image
"""


def BGR2BINARY(image, threshold):
    # convert to gray sclae
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # apply the threshold
    blur = cv2.GaussianBlur(gray_image, (5, 5), 0)
    _, thresh_image = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Negate the image to get a white background and black character
    binary_image = cv2.bitwise_not(thresh_image)

    return binary_image


"""
A function to get the bounding rectange of the binary image
"""


def getBoundingRect(image):
    x1, y1, w, h = cv2.boundingRect(image)
    x2 = x1 + w
    y2 = y1 + h

    bounding_rect_image = image[y1:y2, x1:x2]
    bounding_rect_image = cv2.bitwise_not(bounding_rect_image)
    return bounding_rect_image


frame = cv2.imread('archive/Img/img045-047.png')

binary_image = BGR2BINARY(frame, 0)

boundingBox = getBoundingRect(binary_image)

box_height, box_width = boundingBox.shape
squareSize = box_height if box_height >= box_width else box_width

img = np.zeros((squareSize, squareSize, 3), dtype=np.uint8)
img[:, :] = (255, 255, 255)


cv2.imshow('bounding frame', img)
cv2.imshow('bounding ', boundingBox)


cv2.waitKey(0)
cv2.destroyAllWindows()