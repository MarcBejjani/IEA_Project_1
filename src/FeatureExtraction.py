import csv
from skimage.feature import hog
from GetBoundingRectangle import *
from tqdm import tqdm
from FeaturesPLots import *

def getBlackToWhiteRatio(image):
    blackToWhiteRatio = np.sum(image == 0) / np.sum(image == 255)

    return blackToWhiteRatio


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
        vertical_Symmetry = 1

    return vertical_Symmetry

def getAspectRatio(image):
    x1, y1, w, h = cv2.boundingRect(image)
    aspectRatio = w / h

    return aspectRatio


def normalize_2d(matrix):
    norm = np.linalg.norm(matrix)
    normalized_matrix = matrix/norm

    return normalized_matrix


def getProjectionHistogram(image):

    column_sum = np.sum(image, axis=0)  # sum the values in each column of the img
    row_sum = np.sum(image, axis=1)  # sum the values in each row of the img

    column_sum = normalize_2d(column_sum).flatten()
    row_sum = normalize_2d(row_sum).flatten()

    # dimensions = image.shape #to print
    # x_axis = np.arange(0, dimensions[1]) #to print
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert BGR to RBG since open cv is BGR and matplot is RGB
    # projectionHistogramPlot(image, x_axis, column_sum, row_sum)
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

    x_axis = np.arange(0, dimensions[1])  # creating a matrix of values ranging from 0 till dimension[1] (width of image)
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

    # y_axis = np.arange(0, dimensions[0]) #to plot
    # profilePlot(x_axis, y_axis, bottom, top, left, right)

    return bottom, top, left, right


def getHOG(image):

    resized_img = cv2.resize(image, (128, 64))
    hog_feature, image_hog = hog(resized_img, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(4, 4), visualize=True)  # Set visualize to true if we need to see the image

    return hog_feature


def getListOfCharacters():
    listOfCharacters = []
    excelFile = 'english.csv'
    with open(excelFile, 'r') as data:
        for line in tqdm(csv.DictReader(data)):
            listOfCharacters.append(line)

    return listOfCharacters


def featuresToCSV(lisOfCharacters, directory): #directory of the cropped images

    features = []

    for idx, img in tqdm(enumerate(lisOfCharacters)):
        image_name = lisOfCharacters[idx]['image'] #image name
        image_name = image_name[image_name.index('/')+1:]

        image = cv2.imread(os.path.join(directory, image_name))
        image = BGR2BINARY(image)

        img['BlackToWhite'] = getBlackToWhiteRatio(image)
        img['Horizontal Symmetry'] = horizontalSymmetry(image)
        img['Inverse Symmetry'] = inverseSymmetry(image)
        img['Vertical Symmetry'] = verticalSymmetry(image)
        img['Aspect Ratio'] = getAspectRatio(image)

        #Projection histogram
        column_sum, row_sum = getProjectionHistogram(image)
        for idx, value in enumerate(column_sum):
            img[f'Column Histogram {idx}'] = value

        for idx, value in enumerate(row_sum):
            img[f'Row Histogram {idx}'] = value

        #Profile
        bottom, top, left, right = getProfile(image)
        for idx, value in enumerate(bottom):
            img[f'Bottom Profile {idx}'] = value

        for idx, value in enumerate(top):
            img[f'Top Profile {idx}'] = value

        for idx, value in enumerate(left):
            img[f'Left Profile {idx}'] = value

        for idx, value in enumerate(right):
            img[f'Right Profile {idx}'] = value

        #Hog
        image_hog = getHOG(image)
        for idx, value in enumerate(image_hog):
            img[f'HOG {idx}'] = value

        features.append(img)

        keys = features[0].keys()

    with open('featureDataset.csv', 'w') as csv_file:
        dict_writer = csv.DictWriter(csv_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(features)

def saveToCSV():
    directory = r'C:\Users\leaba\PycharmProjects\IEAproject\Database\croppedImage'  # directory of the processed images
    list_of_Characters = getListOfCharacters()
    featuresToCSV(list_of_Characters, directory)

def main(image):
    chars = getListOfCharacters()
    # getBlackToWhiteRatio(image)
    # horizontalSymmetry(image)
    # verticalSymmetry(image)
    # inverseSymmetry(image)
    # getProfile(image)
    # getProjectionHistogram(image)
    # getHOG(image)
    # getAspectRatio(image)
    # return chars


if __name__ == '__main__':
    #image = cv2.imread('img001-002.png', cv2.IMREAD_UNCHANGED)
    # set = main(image)
    saveToCSV()



