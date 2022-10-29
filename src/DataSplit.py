import pandas as pd
from FeatureExtraction import *
from sklearn.model_selection import train_test_split

"""
Group dataset through the characters 
"""
def groupDataset():

    grouped_data = {}
    list_Of_Characters = getListOfCharacters()

    df = pd.DataFrame(list_Of_Characters)

    groupby_label = df.groupby('label') # grouping the data according to the 'label' --> the different characters

    for groups, data in groupby_label:
        grouped_data[groups] = data

    print(grouped_data)

    return grouped_data # dictionary: keys-> characters / Value-> images corresponding to that character

"""
Split the dta for training and evaluation
"""
def splitDataset(grouped_data):

    X_train = []
    X_test = []
    Y_train = []
    Y_test = []

    for key in tqdm(grouped_data.keys()):
        X = grouped_data[key]['image']
        Y = grouped_data[key]['label']

        x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.8, test_size=0.2, random_state=101)

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


if __name__ == '__main__':
    grouped_data = groupDataset()
    X_train, Y_train, X_test, Y_test = splitDataset(grouped_data)

