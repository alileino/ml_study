import numpy as np
import pandas as pd
import os.path as path

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import cross_val_score, KFold
from measures.cv import cv_accuracy_score, KFold

from sklearn.neighbors import KNeighborsClassifier
from models.knn import KNN

import matplotlib.pyplot as plt
PATH = "../data/KDD_Cup99"
TRAIN_FILE = path.join(PATH, "TrainData.csv")
TEST_FILE = path.join(PATH, "TestData.csv")

'''
For any questions regarding this exercise contact:    parmov@utu.fi

In this exercise the KDDcup99 intrusion detection dataset is given. The actual KDDcup training data set contain 38 different network attack types.

KDDcup99 Dataset: http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html

The data given for this exercise is a simplified version of KDDcup99 containing 5 categories(Classes) of traffic.

Normal class =1

DoS = 2

Probe = 3

R2L = 4

U2R = 5

A training dataset of 5600 different labeled (Last column) network connection data instances for constructing the classifier.

A test data set of 3100 datapoint (different distribution from training data) is given for testing the classifier, data is labeled   (Last column).

Task:

    Use the K-nearest neighbor approach for the classification problem with different number of neighbors (k=3,...,10).
     Performance evaluation utilizing Cross validation on the training set (10-fold) .
     Test the classifier’s accuracy on the test dataset and report the accuracy and F-score for the best number of neighbors (K).
    Include the confusion matrix in your report

Bonus:

    Visualize the training data using a visualization technique (e.g.,Principle component analysis (PCA)) .
    Apply feature selection techniques to choose the features best representing the data.

Notification:

    If the program is slow due to the size of the dataset you can downsize the data, subsample from each 5 class of data, create a smaller dataset containing all 5 classes and perform K-NN on that subset.


'''
class KddDataProvider:
    label_columns = [1,2,3]
    def __init__(self):
        # self.load_data()
        trainX, trainY, testX, testY = self.load_pandas()
        self.trainX = trainX
        self.trainY = trainY
        self.testX = testX
        self.testY = testY

    def get_train(self):
        return self.trainX, self.trainY

    def get_test(self):
        return self.testX, self.testY

    def load_pandas(self):
        dftrain = pd.read_csv(TRAIN_FILE, header=None)

        # dftrain = dftrain.sample(frac=0.3, random_state=1)
        orig_len = len(dftrain)
        dftrain.dropna(axis=1, how="any", inplace=True)
        print("dropped: ", orig_len - len(dftrain), "values")
        train_size = len(dftrain)
        dftest = pd.read_csv(TEST_FILE, header=None)
        dftest = dftest.dropna(axis=0, how="any")
        df = dftrain.append(dftest , ignore_index=True)
        for c in df.columns:
            if len(df[c].unique()) <= 1:
                df = df.drop(c, axis=1)
        for label_column in KddDataProvider.label_columns:
            df[label_column] = df[label_column].factorize()[0]

        trainY = df.values[:train_size,-1]

        testY = df.values[train_size:, -1]
        df = df.drop(len(df.columns), axis=1)

        df = (df - df.mean()) / (df.max() - df.min())

        trainX = df.values[:train_size,:-1]
        testX = df.values[train_size:, :-1]
        return trainX, trainY, testX, testY

    def load_data(self):
        dftrain = pd.read_csv(TRAIN_FILE, header=None)
        dftest = pd.read_csv(TEST_FILE, header=None)

        trainX =  dftrain.values[:, :-1]
        trainY = dftrain.values[:, -1]
        self.test_begin = len(trainX)
        testX = dftest.values[:, :-1]
        testY = dftrain.values[:, -1]
        X = np.array([])
        X = np.concatenate((trainX, testX), 0)
        print("lenTX:", len(trainX), "lenTEX", len(testX), "lenX:", len(X))
        # X = self.preprocess(X)
        print(X[0])

    def preprocess(self, X):
        for label_column in KddDataProvider.label_column:
            label_enc = LabelEncoder()
            column = X[:, label_column]
            missing = np.isnan(column)
            X[missing, label_column] = "-1"

            transform = label_enc.fit_transform(X[:, label_column])
        return X



def ftest(estimator, X, y):
    predy = estimator.predict(X)
    return f1_score(y, predy, average="weighted")

def accuracy(estimator, X, y):
    predy = estimator.predict(X)
    return accuracy_score(y, predy)


def plot_model_selection():
    neighbors = np.arange(1,11)

    data = KddDataProvider()
    X, y = data.get_train()

    scorers = [accuracy, ftest]
    scores = {scorer : [] for scorer in scorers}
    for score_func in scorers:
        for k in neighbors:

            knn = KNeighborsClassifier(n_neighbors=k, weights="uniform")
            # knn = KNN(n_neighbors=k)
            # s = cross_val_score(knn, X, y=y, cv=10, scoring=score_func)
            s = cv_accuracy_score(knn, X, y, cv=KFold(n_splits=10))
            scores[score_func].append(s)


    plt.figure()
    scorer_names = {accuracy:"Accuracy", ftest:"f1-test"}
    offset = 0
    for scorer in scorers:
        allscores = np.array(scores[scorer])

        y = np.mean(allscores, axis=1)

        # Uncomment to plot std error bars
        # e = np.std(allscores, axis=1)

        # plt.errorbar(neighbors+offset, y, yerr=e, lw=2, label=scorer_names[scorer])
        plt.plot(neighbors, y, label=scorer_names[scorer])
        plt.legend()
        offset += 0

    plt.suptitle("10-fold average CV score and Std for KNN")
    plt.xlabel("k (neighbors)")
    plt.ylabel("Score")
    plt.xticks(neighbors)
    plt.show()



plot_model_selection()

def plot_confusion_matrix():
    data = KddDataProvider()
    trainX, trainY = data.get_train()
    testX, testY = data.get_test()
    # knn = KNeighborsClassifier(n_neighbors=3, weights="distance")
    knn = KNN(n_neighbors=9)
    knn.fit(trainX, trainY)
    predY = knn.predict(testX)
    accuracy = accuracy_score(testY, predY)
    fscore = f1_score(testY, predY, average="weighted")
    print("Accuracy:", accuracy, "Fscore", fscore)

plot_confusion_matrix()