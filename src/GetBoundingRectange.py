import cv2
import numpy as np
import os

"""
A function to get the binary black and white image from a RGB image
"""
def BGR2BINARY (image, threshold):
    # convert to gray sclae
    gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    # apply the threshold
    blur = cv2.GaussianBlur(gray_image,(5,5),0)
    _, thresh_image = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    # Negate the image to get a white background and black character
    binary_image = cv2.bitwise_not(thresh_image)
    
    # Apply opening to remove noise
    kernel = np.ones((4,4),np.uint8)
    final_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
    return final_image

"""
A function to get the bounding rectange of the binary image
"""
def getBoundingRect(image):
    x1,y1,w,h = cv2.boundingRect(image)
    x2 = x1+w
    y2 = y1+h

    bounding_rect_image = image [y1:y2,x1:x2]
    bounding_rect_image = cv2.bitwise_not(bounding_rect_image)
    return bounding_rect_image


"""
A function to get the bounding box picture resized to a square
"""


def resizeToSquare(boundingBox):
    box_height, box_width = boundingBox.shape
    if box_height >= box_width:
        img = np.zeros((box_height, box_height, 3), dtype=np.uint8)
        img[:, :] = 255
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sizeDifference = int((box_height - box_width) / 2)
        img[:, sizeDifference:sizeDifference + box_width] = boundingBox
    else:
        img = np.zeros((box_width, box_width, 3), dtype=np.uint8)
        img[:, :] = 255
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sizeDifference = int((box_width - box_height) / 2)
        img[sizeDifference:sizeDifference + box_height, :] = boundingBox
    return img


"""
Method to resize square frame image
"""


def resizeImage(img, height, width):
    return cv2.resize(img, (height, width))


"""
Method that does all the previous steps in one function
"""


def processImage(image, height, width):
    srcImg = cv2.imread(image)
    binaryImage = GetBoundingRectange.BGR2BINARY(srcImg)
    boundingRect = GetBoundingRectange.getBoundingRect(binaryImage)
    squareFrame = resizeToSquare(boundingRect)
    resizedImg = resizeImage(squareFrame, height, width)

    return resizedImg


"""
Save images to directory
"""


def saveImages(dirName):
    if not os.path.exists('./' + dirName):
        os.makedirs('./' + dirName)

    for filename in os.listdir(dirName):
        f = os.path.join(dirName, filename)
        if os.path.isfile(f):
            toAdd = processImage(f, 30, 30)
            cv2.imwrite(f, toAdd)


"""
Main function
"""


def main():
    saveImages('ProcessedImages')


if __name__ == '__main__':
    main()