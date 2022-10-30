import csv
from skimage.feature import hog
from GetBoundingRectange import *
from tqdm import tqdm
import pandas as pd


def getBlackToWhiteRatio(image):
    blackToWhiteRatio = np.sum(image == 0) / (np.sum(image == 255) + 0.00000000000000001)
    return blackToWhiteRatio

def getBlackRatio(image):
    blackRatio = np.sum(image==0)/(np.sum(image == 0) +np.sum(image == 255))
    return blackRatio

def horizontalSymmetry(image):
    blackTop = np.sum(image[0:15, :] == 0)
    blackBottom = np.sum(image[15:30, :] == 0)
    ratio = blackTop / blackBottom
    if 0.8 <= ratio <= 1.2:
        horizontal_Symmetry = 1
    else:
        horizontal_Symmetry = 0

    return horizontal_Symmetry


def inverseSymmetry(image):

    blackTopLeft = np.sum(image[0:15, 0:15] == 0)
    blackTopRight = np.sum(image[0:15, 15:30] == 0)
    blackBottomLeft = np.sum(image[15:30, 0:15] == 0) + 0.000000000000001
    blackBottomRight = np.sum(image[15:30, 15:30] == 0) + 0.000000000000001
    ratio1 = blackTopLeft / blackBottomRight
    ratio2 = blackTopRight / blackBottomLeft

    inverse_Symmetry = 0
    if 0.8 <= ratio1 and ratio2 <= 1.2:
        inverse_Symmetry = 1

    return inverse_Symmetry


def verticalSymmetry(image):
    blackLeft = np.sum(image[:, 0:15] == 0)
    blackRight = np.sum(image[:, 15:30] == 0) + 0.0000000000000000001
    ratio = blackLeft / blackRight
    if 0.9 <= ratio <= 1.1:
        vertical_Symmetry = 1
    else:
        vertical_Symmetry = 0

    return vertical_Symmetry



def getAspectRatio(image):
    w, h = image.shape
    aspectRatio = h / w

    return aspectRatio


def normalize_2d(matrix):
    norm = np.linalg.norm(matrix)
    normalized_matrix = matrix / norm

    return normalized_matrix


def getProjectionHistogram(image):
    column_sum = np.sum(image, axis=0)  # sum the values in each column of the img
    row_sum = np.sum(image, axis=1)  # sum the values in each row of the img

    column_sum = normalize_2d(column_sum).flatten()
    row_sum = normalize_2d(row_sum).flatten()

    return column_sum, row_sum


def getProfile(image):
    dimensions = image.shape  # (m,n) m:rows/ n:columns

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
    if np.sum(bottom) != 0:
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
    if np.sum(top) != 0:
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
    if np.sum(right) != 0:
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
    if np.sum(left) != 0:
        left = normalize_2d(left)


    return bottom, top, left, right


def getHOG(image):
    resized_img = cv2.resize(image, (128, 64))
    hog_feature, image_hog = hog(resized_img, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(4, 4),
                                 visualize=True)  # Set visualize to true if we need to see the image

    return hog_feature


def getListOfCharacters():
    listOfCharacters = []
    excelFile = 'english.csv'
    with open(excelFile, 'r') as data:
        for line in tqdm(csv.DictReader(data)):
            listOfCharacters.append(line)

    return listOfCharacters


def featuresToCSV(lisOfCharacters):  # directory of the cropped images
    dirname, filename = os.path.split(os.path.abspath(__file__))
    
    features = []

    for idx, img in tqdm(enumerate(lisOfCharacters)):

        image_name = lisOfCharacters[idx]['image']  # image name
        image_name = image_name[image_name.index('/') + 1:]

        dirResizedWhite = dirname+f'\SquaredWithWhiteAdded\{image_name}'
        dirResizedNoWhite = dirname+f'\SquaredWithNoWhiteAdded\{image_name}'
        dirBounding = dirname+f'\BoundingBoxes\{image_name}'

        boundingRectangleImage = cv2.imread(dirBounding)
        boundingRectangleImage = cv2.cvtColor(boundingRectangleImage,cv2.COLOR_BGR2GRAY)

        ResizedBoundingWithWhite = cv2.imread(dirResizedWhite)
        ResizedBoundingWithWhite = cv2.cvtColor(ResizedBoundingWithWhite,cv2.COLOR_BGR2GRAY)

        ResizedBoundingWithNoWhite = cv2.imread(dirResizedNoWhite)
        ResizedBoundingWithNoWhite = cv2.cvtColor(ResizedBoundingWithNoWhite,cv2.COLOR_BGR2GRAY)

        img['BlackToWhite'] = getBlackToWhiteRatio(boundingRectangleImage)
        img['Horizontal Symmetry'] = horizontalSymmetry(boundingRectangleImage)
        img['Inverse Symmetry'] = inverseSymmetry(boundingRectangleImage)
        img['Vertical Symmetry'] = verticalSymmetry(boundingRectangleImage)
        img['Aspect Ratio'] = getAspectRatio(boundingRectangleImage)

        # Projection histogram
        column_sum, row_sum = getProjectionHistogram(ResizedBoundingWithNoWhite)
        for idx1, value in enumerate(column_sum):
            img[f'Column Histogram {idx1}'] = value

        for idx1, value in enumerate(row_sum):
            img[f'Row Histogram {idx1}'] = value

        # Profile
        bottom, top, left, right = getProfile(ResizedBoundingWithNoWhite)
        for idx1, value in enumerate(bottom):
            img[f'Bottom Profile {idx1}'] = value

        for idx1, value in enumerate(top):
            img[f'Top Profile {idx1}'] = value

        for idx1, value in enumerate(left):
            img[f'Left Profile {idx1}'] = value

        for idx1, value in enumerate(right):
            img[f'Right Profile {idx1}'] = value

        # Hog
        image_hog = getHOG(ResizedBoundingWithNoWhite)
        for idx1, value in enumerate(image_hog):
            img[f'HOG {idx1}'] = value

        features.append(img)

        keys = features[0].keys()
    df = pd.DataFrame(features)
    df.to_csv('FeaturesWithNoWhite.csv', index=False, header=True)


def saveToCSV():
    list_of_Characters = getListOfCharacters()
    featuresToCSV(list_of_Characters)


if __name__ == '__main__':
    saveToCSV()
    