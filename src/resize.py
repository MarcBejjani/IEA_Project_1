import cv2
import numpy as np
import os
import GetBoundingRectange

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
    if not os.path.exists('./'+dirName):
        os.makedirs('./'+dirName)
        
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
