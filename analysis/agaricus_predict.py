import numpy as np
import pandas as pd
from models.knn import KNN
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

from sklearn.neighbors import KNeighborsClassifier

from measures.cv import KFold, cv_accuracy_score, cv_squares_score
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing.label import LabelBinarizer

import matplotlib.pyplot as plt

def get_agaricus_data(row_limit=None):
    df = pd.read_csv("data/agaricus-lepiota.data")
    # Randomize the rows because they are ordered
    df = df.sample(frac=1, random_state=1)
    X = df.values[:, 1:]

    y = df.values[:, 0]
    enc = [MultiLabelBinarizer() for _ in range(len(X[0]))]
    newX = np.empty((len(X),0), dtype=np.float64)
    for i, encoder in enumerate(enc):
        columns = encoder.fit_transform(X[:, i])
        newX = np.concatenate((newX, columns), axis=1)
    X = newX
    y = LabelBinarizer().fit_transform(y)
    y = y.flatten()

    if row_limit:
        X = X[:row_limit, :]
        y = y[:row_limit]
    return X,y

def test_knn(row_limit=None, n_neighbors=5):
    X, y = get_agaricus_data(row_limit=row_limit)
    models = (KNN(n_neighbors=n_neighbors), KNeighborsClassifier(n_neighbors=n_neighbors))
    names = ("Custom", "Scikit-learn")
    scores = []
    for model, name in zip(models, names):
        score = cv_accuracy_score(model, X, y)
        scores.append(score)
    diffs = []
    for score in scores:
        diffs.append(scores[0]-score)
    return list(zip(names, scores, diffs))

def test_multiple_neighbors(row_limit=None, regression=False, max_neighbors=10):
    X,y = get_agaricus_data(row_limit=row_limit)
    n_neighbors = np.arange(1,max_neighbors+1)
    score_func = cv_accuracy_score
    if regression:
        score_func = cv_squares_score
    scores = []
    for k in n_neighbors:
        model = KNN(n_neighbors=k, regression=False)

        score = score_func(model, X, y)
        scores.append(score)
    return (n_neighbors, scores)

def plot_neighbors(regression=False):
    plt.figure()
    n_neighbors, scores = test_multiple_neighbors(row_limit=200, regression=regression)
    plt.plot(n_neighbors,scores)
    ylabel = "accuracy score"
    type = "classification"
    if regression:
        ylabel = "squared difference"
        type = "regression"
    plt.suptitle("KNN CV mean vs. neighbors (%s)" % type)
    plt.xlabel("neighbors")
    plt.ylabel(ylabel)
    plt.savefig("agaricus_neighbors_" + str(regression) + ".png")


def help_print(scorelist):
    print("\n".join(["%s: %.5f (diff: %+.5f)" % (m, s, d) for m, s, d in scorelist]))

def plot_everything():
    plot_neighbors(regression=True)

    plot_neighbors(regression=False)
    print("For n=100 and n_neighbors=5")
    results = test_knn(row_limit=100, n_neighbors=5)
    help_print(results)
