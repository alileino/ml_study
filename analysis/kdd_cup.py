import numpy as np
import pandas as pd
import os.path as path

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from measures.cv import cv_accuracy_score, KFold

from models.knn import KNN
from helpers import plot_confusion_matrix

import matplotlib.pyplot as plt

PATH = "../data/KDD_Cup99"
TRAIN_FILE = path.join(PATH, "TrainData.csv")
TEST_FILE = path.join(PATH, "TestData.csv")

'''

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

OUTPUT_PATH = "../reports/img"

class KddDataProvider:
    '''
    Loads and preprocesses KddCup99 data.
    '''
    label_columns = [1,2,3]
    def __init__(self, debug=False):
        '''
        :param debug: If true, it uses a small part of the entire training data set
        '''
        trainX, trainY, testX, testY = self.__load_data(debug)
        self.trainX = trainX
        self.trainY = trainY
        self.testX = testX
        self.testY = testY

    def get_train(self):

        return self.trainX, self.trainY

    def get_test(self):
        return self.testX, self.testY

    def __load_data(self, debug):
        '''
        Loads the training and test data and does preprocessing to them
        :param debug:
        :return:
        '''
        dftrain = pd.read_csv(TRAIN_FILE, header=None, names=np.arange(0,42))

        if debug:
            dftrain = dftrain.sample(frac=0.3, random_state=1)

        train_size = len(dftrain)
        dftest = pd.read_csv(TEST_FILE, header=None, names=np.arange(0,42))

        # Drop rows with NA's
        dftest = dftest.dropna(axis=0, how="any")
        df = dftrain.append(dftest , ignore_index=True)

        for c in df.columns: # Drop columns with only a single value,
            if len(df[c].unique()) <= 1:
                df = df.drop(c, axis=1)
        df = self.__encode(df)


        trainY = df.values[:train_size,-1]

        testY = df.values[train_size:, -1]
        df = df.drop(df.columns[-1], axis=1) # drop the last column

        # Z-score normalize the data
        df = (df - df.mean()) / (df.max() - df.min())

        trainX = df.values[:train_size,:-1]
        testX = df.values[train_size:, :-1]
        return trainX, trainY, testX, testY

    def __encode(self, df):
        newdf = df[df.columns[0]]

        for label_column in KddDataProvider.label_columns:
            newdf = pd.concat((newdf, pd.get_dummies(df[label_column], '', '').astype(int)), axis=1, ignore_index=True)

        newdf = pd.concat((newdf, df[df.columns[4:]]), axis=1)
        newdf.columns = np.arange(0, len(newdf.columns))

        return newdf

def plot_model_selection():
    neighbors = np.arange(1,11)

    data = KddDataProvider()
    X, y = data.get_train()

    scores = []

    for k in neighbors:

        knn = KNN(n_neighbors=k)
        s = cv_accuracy_score(knn, X, y, cv=KFold(n_splits=10))
        scores.append(s)


    plt.figure()
    y = np.mean(scores, axis=1)

    # Uncomment to plot std error bars
    # e = np.std(allscores, axis=1)

    # plt.errorbar(neighbors+offset, y, yerr=e, lw=2, label=scorer_names[scorer])
    plt.plot(neighbors, y, label="Accuracy")
    plt.legend()

    plt.suptitle("10-fold average CV score and Std for KNN")
    plt.xlabel("k (neighbors)")
    plt.ylabel("Score")
    plt.xticks(neighbors)
    plt.savefig(path.join(OUTPUT_PATH, "kdd_model_selection.png"), format="PNG")



# plot_model_selection()

def plot_confusion(truey, predy):
    plt.figure()
    cm = confusion_matrix(truey, predy)
    plot_confusion_matrix(cm, np.unique(truey))
    plt.savefig(path.join(OUTPUT_PATH, "kdd_confusion_matrix.png"), format="PNG")


def print_results():
    data = KddDataProvider(debug=False)
    trainX, trainY = data.get_train()
    testX, testY = data.get_test()
    knn = KNN(n_neighbors=8)
    knn.fit(trainX, trainY)
    predY = knn.predict(testX)
    accuracy = accuracy_score(testY, predY)
    fscore = f1_score(testY, predY, average="weighted")
    print("Accuracy:", accuracy, "F1-score", fscore)
    plot_confusion(testY, predY)
print_results()