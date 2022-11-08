import pandas as pd
from src.Website.FeatureExtraction import *
from sklearn.model_selection import train_test_split
import itertools


def getListOfCharacters():
    listOfCharacters = []
    excelFile = 'english.csv'
    with open(excelFile, 'r') as data:
        for line in tqdm(csv.DictReader(data)):
            listOfCharacters.append(line)

    return listOfCharacters


"""
Group dataset through the characters 
"""


def groupDataset(): # Group the data according to the labels in a list

    grouped_data = {}
    list_Of_Characters = getListOfCharacters()
    df = pd.DataFrame(list_Of_Characters)
    groupby_label = df.groupby('label') # grouping the data according to the 'label' --> the different characters

    for groups, data in groupby_label:
        grouped_data[groups] = data
    print(grouped_data)
    return grouped_data # dictionary: keys-> characters / Value-> images corresponding to that character


"""
Split the data for training and evaluation
"""


def splitDataset(grouped_data, RS): # Split the data 80/20 by taking 80% of the images for each character

    X_train = []
    X_test = []
    Y_train = []
    Y_test = []

    for key in tqdm(grouped_data.keys()):
        X = grouped_data[key]['image']
        Y = grouped_data[key]['label']

        x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.8, test_size=0.2, random_state=RS)

        X_train.append(x_train)
        X_test.append(x_test)
        Y_train.append(y_train)
        Y_test.append(y_test)

    #Convert the elements in X_train... from pandas series to list
    for idx, val in enumerate(X_train):
        X_train[idx] = X_train[idx].tolist()
    for idx, val in enumerate(Y_train):
        Y_train[idx] = Y_train[idx].tolist()
    for idx, val in enumerate(X_test):
        X_test[idx] = X_test[idx].tolist()
    for idx, val in enumerate(Y_test):
        Y_test[idx] = Y_test[idx].tolist()

    return X_train, Y_train, X_test, Y_test


# Splitting the data
def getFeaturesWithWhite(X_train, X_test): # Get the features corresponding to each image of the X_train and X_test

    X_train_features = []
    X_test_features = []

    df = pd.read_csv('FeaturesWithWhite.csv')
    df = df.drop(['label', 'Horizontal Symmetry', 'Inverse Symmetry', 'Vertical Symmetry'], axis=1)

    # Get X_train_features: returns a list of lists including images and their corresponding features
    for character in range(62): # len(X_train) = 62 | len(X_train[0]) = 44 (corresponding to 44 images for character 0
        for image in range(44): # 44: Since it is a 80/20 split
            image_number = X_train[character][image]
            features = df.loc[df['image'] == image_number]  # Getting the features(row) corresponding to the image name
            features = features.drop(['image'], axis=1)
            features = features.values.tolist()
            merged = list(itertools.chain.from_iterable(features))
            X_train_features.append(merged)

    # Get X_test_features
    for character in range(62):
        for image in range(11):
            image_number = X_test[character][image]
            features = df.loc[df['image'] == image_number]  # Getting the features corresponding to the image name
            features = features.drop(['image'], axis=1)
            features = features.values.tolist()
            merged = list(itertools.chain.from_iterable(features))
            X_test_features.append(merged)

    return X_train_features, X_test_features

def getFeaturesNoWhite(X_train, X_test):

    X_train_features = []
    X_test_features = []

    df = pd.read_csv('FeaturesNoWhite.csv')
    df = df.drop(['label', 'Horizontal Symmetry', 'Inverse Symmetry', 'Vertical Symmetry'], axis=1)

    # Get X_train_features: returns a list of lists including images and their corresponding features
    for character in range(62):
        for image in range(44):
            image_number = X_train[character][image]
            features = df.loc[df['image'] == image_number]  # Getting the features(row) corresponding to the image name
            features = features.drop(['image'], axis = 1)
            features = features.values.tolist()
            merged = list(itertools.chain.from_iterable(features))
            X_train_features.append(merged)

    # Get X_test_features
    for character in range(62):
        for image in range(11):
            image_number = X_test[character][image]
            features = df.loc[df['image'] == image_number]  # Getting the features corresponding to the image name
            features = features.drop(['image'], axis=1)
            features = features.values.tolist()
            merged = list(itertools.chain.from_iterable(features))
            X_test_features.append(merged)

    return X_train_features, X_test_features


if __name__ == '__main__':
    grouped_data = groupDataset()
    # X_train, Y_train, X_test, Y_test = splitDataset(grouped_data)

