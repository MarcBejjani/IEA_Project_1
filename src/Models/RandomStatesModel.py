import random
import joblib
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from DataSplit import *


def parameterTuning(X_train_features, Y_train, RS):
    hyper_param = {'C': [0.1, 1, 10], 'gamma': [1, 0.1, 0.01], 'kernel': ['linear', 'rbf', 'poly']}

    gs = GridSearchCV(estimator=SVC(),
                       param_grid=hyper_param,
                       cv=StratifiedKFold(5).split(X_train_features, Y_train),
                       scoring='accuracy',
                       refit=True,
                       verbose=3)

    gs.fit(X_train_features, Y_train)
    print(type(gs))

    # save the model to disk
    joblib.dump(gs.best_estimator_, f'Model_WithWhite{RS}.npy')

    return gs


def prediction(X_test_features, Y_test, RS):

    model = {'Random State': [RS]}
    df = pd.DataFrame(model)
    df.to_csv('ModelWithWhite.csv', mode='a', index=False)

    classifier = joblib.load(f'Model_WithWhite{RS}.npy')
    y_pred = classifier.predict(X_test_features)

    # metrics
    # print("accuracy", metrics.accuracy_score(Y_test, y_pred), "\n")
    print(metrics.confusion_matrix(Y_test, y_pred), "\n")

    clsf_report = pd.DataFrame(classification_report(y_true=Y_test, y_pred=y_pred, output_dict=True)).transpose()
    clsf_report.to_csv(f'ModelWithWhite.csv', mode='a', index=True)


def randomList(): #Generating a random list to test the models

    rn_list = []

    for rs in range(20):
        rn = random.randint(30,150)
        for i in rn_list:
            while (i == rn) or (i == 100) or (i == 101) or (i == 105) or (i == 41):
                rn = random.randint(30,150)

        rn_list.append(rn)

    return rn_list


def modelsDifferentRS (rn_list): #Going through elts of the random list

    grouped_data = groupDataset()
    for RS in tqdm(rn_list):

        X_train, Y_train, X_test, Y_test = splitDataset(grouped_data, RS=RS) #Add a parameter for splitData
        X_train_features, X_test_features = getFeaturesWithWhite(X_train, X_test) # Testing with no white pixels

        Y_train = list(itertools.chain.from_iterable(Y_train))
        Y_test = list(itertools.chain.from_iterable(Y_test))

        classifier = parameterTuning(X_train_features, Y_train, RS)
        prediction( X_test_features, Y_test, f'{RS}')


if __name__ == '__main__':
    rn_list = randomList()
    modelsDifferentRS(rn_list)




