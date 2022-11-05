import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split

from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

from sklearn.decomposition import PCA

from ImageToFeature import getFeatures

#import file and reading few lines
# data = pd.read_csv('dataSet.csv')
#data = pd.read_csv('FeaturesWithWhite.csv')
# data = pd.read_csv('../NewFeatureSet.csv')

# data.loc[0]
#
#
# cnames = list(data.columns)
#
# cnames = list(data.columns)
#
# ndata = data
#
#
# # remove Symmetry
# for c in cnames:
#   if c.find('Symmetry') != -1:
#     ndata = ndata.drop(c,axis =1)
#
#
#
# y = data['label']
# ndata = ndata.drop("label", axis = 1)
# ndata = ndata.drop("image", axis = 1)


# X = ndata
# train test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7,test_size = 0.3, random_state = 101)
# pca = PCA(n_components=50)
# X_train = pca.fit_transform(X_train)
# X_test = pca.transform(X_test)

"""MODEL BUILDING"""

# model_linear = SVC(kernel='linear', probability = True)
# model_linear.fit(X_train, y_train)
savedSVM = 'savedSVM.sav'
# joblib.dump(model_linear, savedSVM)

# predict
model_linear = joblib.load(savedSVM)
# y_pred = model_linear.predict(X_test)

userInput = getFeatures('static/uploads/img.png')
y_new = model_linear.predict(userInput)
print(y_new)
# accuracy
# print("accuracy:", metrics.accuracy_score(y_true=y_test, y_pred=y_pred), "\n")
