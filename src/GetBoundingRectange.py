import cv2
import numpy as np
import os
import csv

"""
A function to get the binary black and white image from a RGB image
"""


def BGR2BINARY(image, x, y):
    # convert to gray scale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # apply the threshold
    blur = cv2.GaussianBlur(gray_image, (5, 5), 0)
    _, thresh_image = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Negate the image to get a black background and white character
    binary_image = cv2.bitwise_not(thresh_image)
    # Apply opening to remove noise
    kernel = np.ones((x, y), np.uint8)
    final_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)
    return final_image


"""
A function to get the bounding rectange of the binary image
"""


def getBoundingRect(image):
    x1, y1, w, h = cv2.boundingRect(image)

    x2 = x1 + w
    y2 = y1 + h
    bounding_rect_image = image[y1:y2, x1:x2]

    # Return image to white background with
    bounding_rect_image = cv2.bitwise_not(bounding_rect_image)
    return bounding_rect_image


def getCountourRect(image):
    countours, hir = cv2.findContours(image, 1, cv2.CHAIN_APPROX_SIMPLE)
    print('number of count: ' + str(len(countours)))
    if len(countours) > 1:
        countours, _ = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        h = 0
        img = 0
        for c in countours:
            x, y, w, h_temp = cv2.boundingRect(c)
            if h_temp > h:
                h = h_temp
                img = c
        x, y, w, h = cv2.boundingRect(img)
        x2 = x + w
        y2 = y + h
        bounding_rect_image = image[y:y2, x:x2]
        bounding_rect_image = cv2.bitwise_not(bounding_rect_image)
        return bounding_rect_image
    else:
        return getBoundingRect(image)


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


def processImage(dir):
    n = dir.find('img')
    srcImg = cv2.imread(dir)
    if dir[n + 3:n + 6] == '045' or dir[n + 3:n + 6] == '046':
        print('i or j')
        binaryImage = BGR2BINARY(srcImg, 10, 10)
        boundingRect = getBoundingRect(binaryImage)
    else:
        print('not i neither j')
        binaryImage = BGR2BINARY(srcImg, 10, 10)
        boundingRect = getCountourRect(binaryImage)

    return boundingRect


"""
Save images to directory
"""


def saveImages(dirName):
    for filename in os.listdir(dirName):
        f = os.path.join('./' + dirName, filename)
        if os.path.isfile(f):
            toAdd = processImage(f, 30, 30)
            cv2.imwrite(f, toAdd)


def getListOfCharacters():
    listOfCharacters = []
    excelFile = 'english.csv'
    with open(excelFile, 'r') as data:
        for line in csv.DictReader(data):
            listOfCharacters.append(line)

    return listOfCharacters


"""
Main function
"""


def main():
    dirname, filename = os.path.split(os.path.abspath(__file__))
    pass


if __name__ == '__main__':
    dirname, filename = os.path.split(os.path.abspath(__file__))
    list_of_Characters = getListOfCharacters()

    for idx, img in enumerate(list_of_Characters):
        print(idx)
        image_name = list_of_Characters[idx]['image']  # image name
        image_name = image_name[image_name.index('/') + 1:]
        if os.path.exists(dirname + f'\EnglishHandwrittenCharacters\{image_name}'):
            # print(image_name)
            dir = dirname + f'\EnglishHandwrittenCharacters\{image_name}'
            boundingRect = processImage(dir)
            # boundingRect = cv2.resize(boundingRect, (30,30))
            boundingRect = resizeToSquare(boundingRect)
            save_dir = dirname + f'\SquaredWithWhiteAdded\{image_name}'
            cv2.imwrite(save_dir, boundingRect)

    # image_name = list_of_Characters[65]['image']
    # print(image_name)
    # image_name = image_name[image_name.index('/') + 1:]
    # dir= dirname+f'\EnglishHandwrittenCharacters\{image_name}'
    # boundingRect = processUserImage(dir)
    # boundingRect = resizeToSquare(boundingRect)
    # boundingRect = cv2.resize(boundingRect, (30,30))
    # print(boundingRect.shape)
    # cv2.imshow('BoundingRect',boundingRect)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # save_dir = dirname+f'\SquaredWithWhiteAdded\{image_name}'
    # print(save_dir)
    # cv2.imwrite(save_dir, boundingRect)

    # boundingRect1 = cv2.resize(boundingRect, (30,30))
    # print(boundingRect[4])
    # print(np.sum(boundingRect==0))
    # cv2.imshow('frame',boundingRect1)
    # boundingRect2 = resizeToSquare(boundingRect)
    # boundingRect2 = cv2.resize(boundingRect2, (30,30))
    # cv2.imshow('frame2', boundingRect2)
    # cv2.waitKey(0)
    # blackToWhiteRatio = np.sum(boundingRect == 0) / (np.sum(boundingRect == 255) + 0.00000000000000001)
    # blackRatio = np.sum(boundingRect == 0) / (np.sum(boundingRect == 255) + 0.00000000000000001)
    # print(np.count_nonzero(boundingRect))
    # print(blackToWhiteRatio)

    # boundingRectangleImage = cv2.imread(dirname+f'\BoundingBoxes\{image_name}')
    # boundingRectangleImage = cv2.cvtColor(boundingRectangleImage,cv2.COLOR_BGR2GRAY)
    # cv2.imshow('frame',boundingRectangleImage)
    # cv2.waitKey(0)
    # print(boundingRectangleImage.shape)
