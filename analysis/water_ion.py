import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from measures.cv import cv_squares_score, KFold, cv_score, aggregate_cv
from measures.distance import c_index
from models.knn import KNN

'''
Third Exercise: Prediction of metal ion content from multi-parameter data

For any questions regarding this exercise contact: iimope@utu.fi

The Water_data.csv file is a multi-parameter dataset consisting of 201 samples obtained from 67 mixtures of Cadmium, Lead, and tap water.  Three features (attributes) were measured for each samples (Mod1, Mod2, Mod3).

Tasks

Use K-Nearest Neighbor Regression to predict total metal concentration (c_total), concentration of Cadmium (Cd) and concentration of Lead (Pb), for each sample.

    The data should be normalized using z-score.
    Implement Leave-One-Out Cross Validation approach and calculate the C-index for each output (c-total, Cd, Pb).
     Implement Leave-Three-Out Cross Validation and calculate the C-index for each output (c-total, Cd, Pb).  This mean to leave out as test set the 3 consecutive samples at the same time (see lecture PowerPoint presentation for details).
    Try different number of neighbors (k= 1, 2, 3, 4, 5) in each approach to find the optimal k.

Return

 A report that includes the results for each approach, and the code used.

'''

class WaterDataProvider:
    objective_columns = ["c_total", "Cd", "Pb"]
    def __init__(self):
        self.X, self.Y, self.df = self.load_data()

    def load_data(self):
        df = pd.read_csv("../data/Water_data.csv")
        y = df[WaterDataProvider.objective_columns]

        X = df.drop(WaterDataProvider.objective_columns, axis=1)
        X = (X - X.mean())/(X.std())
        return X, y, df

def scatters(y):
    data = WaterDataProvider()
    objective = WaterDataProvider.objective_columns
    objective.remove(y)
    df = data.df.drop(objective, axis=1)
    sns.set(style="ticks")
    sns.pairplot(df, hue=y, palette="hls")
    plt.show()

def mean_regression_plot(xticks, y, name):
    MSE = np.mean((y-np.mean(y))**2)+np.random.rand()*10**5
    MSE = np.repeat(MSE, len(xticks))
    plt.plot(xticks, MSE, label=(name + " mean"))

def mean_cindex_plot(xticks, y, name):
    predy = np.repeat(np.mean(y), len(y))
    cindex = c_index(y, predy)
    cindex = np.repeat(cindex, len(xticks))
    plt.plot(xticks, cindex, label=(name + " mean"))


# plot_per_column()

def plot_per_column(leave_out):
    plt.figure()
    data = WaterDataProvider()
    X = data.X.values
    Y = data.Y
    neighbors = np.arange(1, 20)
    for ycolumn in Y.columns:
        y = Y[ycolumn].values
        scores = []
        for k in neighbors:
            knn = KNN(n_neighbors=k, regression=True)
            s = cv_squares_score(knn, X, y, cv=KFold(n_splits=(len(X)//leave_out)))
            scores.append(np.mean(s))

        plt.plot(neighbors, scores, label=ycolumn)
        mean_regression_plot(neighbors, y, ycolumn)

    plt.suptitle("3-Fold CV MSE vs. neighbors")
    plt.ylabel("CV MSE")
    plt.legend()
    plt.xticks(neighbors)
    plt.xlabel("neighbors")
    savefig("water_cindex_plots")

def water_cindex_plots(leave_out, randomize=0, llim=1, hlim=15, show_mean=True):
    '''
    :param leave_out: how many consequtive samples to leave to test set in each K-fold
    :param randomize: float describing the amount of randomness added to line plots
    :param llim: the inclusive lower limit for neighbor count
    :param hlim: the exxclusive higher limit for neighbor count
    :param show_mean: True if the (baseline) mean c-index should be plotted
    :return: None
    '''
    plt.figure()
    data = WaterDataProvider()
    X = data.X.values
    Y = data.Y
    neighbors = np.arange(llim, hlim)
    for ycolumn in Y.columns:
        y = Y[ycolumn].values
        scores = []
        for k in neighbors:
            knn = KNN(n_neighbors=k, regression=True, weights=None)

            s = aggregate_cv(knn, X, y, cv=KFold(n_splits=(len(X)//leave_out)), score_func=c_index)
            scores.append(np.mean(s) + np.random.rand()*randomize)
        plt.plot(neighbors, scores, label=ycolumn)
    if show_mean:
        plt.plot(neighbors, np.repeat(0.5, len(neighbors)))
    plt.suptitle("Leave-%i-Out CV c-index vs. neighbors" % leave_out)
    plt.ylabel("C-index")
    plt.legend()
    plt.xticks(neighbors)
    plt.xlabel("neighbors")
    savefig("water_cindex_plots%i%i" % (leave_out, llim))

def water_cindex_weights(leave_out, llim=1, hlim=11, randomize=0, show_mean=False):
    plt.figure()
    data = WaterDataProvider()
    X = data.X.values
    Y = data.Y
    neighbors = np.arange(llim, hlim)
    ycolumn = "c_total"
    y = Y[ycolumn].values

    for weights in KNN.WEIGHTS:
        scores = []

        for k in neighbors:

            subscores = []
            for ycolumn in Y.columns:

                y = Y[ycolumn].values

                knn = KNN(n_neighbors=k, regression=True, weights=weights)

                s = aggregate_cv(knn, X, y, cv=KFold(n_splits=(len(X)//leave_out)), score_func=c_index)
                subscores.append(np.mean(s))
            scores.append(np.mean(subscores))
        plt.plot(neighbors, scores, label=weights)

    if show_mean:
        plt.plot(neighbors, np.repeat(0.5, len(neighbors)))
    plt.suptitle("Leave-%i-Out CV c-index vs. neighbors" % leave_out)
    plt.ylabel("C-index")
    plt.legend()
    plt.xticks(neighbors)
    plt.xlabel("neighbors")
    savefig("water_cindex_plots_weights%i%i" % (leave_out, llim))

def savefig(name):
    import os.path as path
    IMG_PATH = "../reports/img"
    plt.savefig(path.join(IMG_PATH, name + ".png"), format="png")
    plt.show()
water_cindex_weights(3, llim=2, hlim=25)
# water_cindex_plots(1)
# water_cindex_plots(3, llim=2, hlim=11, show_mean=False)
