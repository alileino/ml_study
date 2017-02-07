import numpy as np
import matplotlib.pyplot as plt
import vis_helpers as vis
from models.knn import KNN
from measures.distance import  l2_distance_squared, c_index
from measures.cv import *
import os.path as path

import pandas as pd
import seaborn as sns

'''
ASSIGNMENT DETAILS

    Evaluate the performance of 5-nearest neighbors algorithm by using the spatial leave-one-out cross validation. Use C-index to the measure the performance.
    Use dead zone radiuses from 0 - 200 meters with 10 meter intervals, i.e. d = 0m, d = 10m, ..., d = 200m.
    Make a plot of the prediction performance (C-index) as the function of dead zone radius. Scale the y-axis to [0, 1] range.
    Remember to normalize the predictor features (INPUT.csv).
    COORDINATES.csv contains the geographical coordinates of the data points, INPUT.csv contains the predictor features and OUTPUT.csv the response variable, i.e. the water permeability exponent.
    More details can be found from the attached lecture slides.

'''

DATA_DIR = "../data/perm"
class WaterPermDataProvider:

    def __init__(self):
        input_path = path.join(DATA_DIR, "INPUT.csv")
        output_path = path.join(DATA_DIR, "OUTPUT.csv")
        coord_path = path.join(DATA_DIR, "COORDINATES.csv")
        X = np.genfromtxt(input_path, delimiter=",")
        X = (X-np.mean(X, axis=0))/np.std(X, axis=0)

        y = np.genfromtxt(output_path, delimiter=",")
        C = np.genfromtxt(coord_path, delimiter=",")
        # TODO: REMEMBER, this is squared
        self.dists = l2_distance_squared(C,C)
        self.coords = C
        self.X, self.y = X, y


def plot_deltas(dmax=11):
    data = WaterPermDataProvider()
    knn = KNN(n_neighbors=5, regression=True, weights=None)
    deltas = (np.arange(1,dmax)*10.)**2
    scores = []
    for delta in deltas:

        cv = SLOO(data.dists, delta=delta)

        s = aggregate_cv(knn, data.X, data.y, cv=cv, score_func=c_index)
        scores.append(s)

    plt.figure()
    plt.plot(np.sqrt(deltas), scores)
    plt.xticks(np.sqrt(deltas))

    plt.xlabel("$\delta$")
    plt.ylabel("c-index")
    plt.legend()
    vis.plot_show("perm_deltas_plot")

def plot_weights(delta=100, kmin=5, kmax=10):
    data = WaterPermDataProvider()
    neighbors =  np.arange(kmin, kmax)
    all_weights = ["ranks", "distance", "none", "inv_distance"]
    # all_weights = [None]
    for weights in all_weights:

        scores = []
        for k in neighbors:

            knn = KNN(n_neighbors=k, regression=True, weights=weights)
            cv = SLOO(data.dists, delta=float(delta**2))

            s = aggregate_cv(knn, data.X, data.y, cv=cv, score_func=c_index)
            scores.append(s)
        plt.plot(neighbors, scores, label=weights)

    plt.xticks(neighbors)
    plt.legend()
    vis.plot_show("perm_weights_plot%i%i" % (kmin, kmax))

def location_error_plot(kind="error"):
    data = WaterPermDataProvider()
    knn = KNN(n_neighbors=5, regression=True)
    cv = SLOO(data.dists, delta=150.)
    s, preds = cv_score(knn, data.X, data.y, cv=cv, score_func=mean_squared_error, return_predictions=True)

    C = data.coords
    C = (C-np.mean(C, axis=0))
    x1,x2 = C[:,0], C[:,1]
    if kind=="error":
        y = (data.y-preds.flatten())**2
    else:
        y = data.y

    # sns.set_style("white")
    hist, bins = np.histogram(y, bins=8)
    y = np.digitize(y, bins=bins)
    y = bins[y-1]

    df = pd.DataFrame(np.array([x1,x2,y]).transpose(), columns=["x1","x2","y"])

    jitter = 0.01*(np.max(C,axis=0)-np.min(C,axis=0))
    x2jitter = x2 + (np.random.rand(*x2.shape)-0.5)*jitter[1]
    x1jitter = x1 + (np.random.rand(*x1.shape)-0.5)*jitter[0]
    # sns.set_context("paper")
    if kind=="error":
        sns.lmplot("x1","x2", data=df, scatter=True, fit_reg=False,hue="y", palette="Reds", scatter_kws={"s":100, }, size=6, aspect=2)

        plt.scatter(x1jitter, x2jitter, c=[0,0,0], s=2)
    else:
        xi, yi = np.linspace(x1.min(), x1.max(), 300), np.linspace(x2.min(), x2.max(), 300)
        xi, yi = np.meshgrid(xi, yi)
        import scipy
        # Interpolate; there's also method='cubic' for 2-D data such as here
        zi = scipy.interpolate.griddata((x1, x2), y, (xi, yi), method='cubic')
        # sns.interactplot("x1", "x2", "y", data=df, levels=40)
        plt.imshow(zi, vmin=y.min(), vmax=y.max(), origin='lower',
                   extent=[x1.min(), x1.max(), x2.min(), x2.max()], cmap="rainbow")
        plt.colorbar()
    plt.suptitle("Squared error in map, delta=150, k=5")
    vis.plot_show("perm_SE_locations")

def permability_plot(plot_error=False):
    data = WaterPermDataProvider()
    if plot_error:
        knn = KNN(n_neighbors=5, regression=True)
        cv = SLOO(data.dists, delta=100.)
        s, preds = cv_score(knn, data.X, data.y, cv=cv, score_func=mean_squared_error, return_predictions=True)

        y = (data.y - preds.flatten()) ** 2
    else:
        y = data.y
    C = data.coords
    C = (C-np.mean(C, axis=0))
    x1,x2 = C[:,0], C[:,1]
    # sns.set_style("white")

    xi, yi = np.linspace(x1.min(), x1.max(), 300), np.linspace(x2.min(), x2.max(), 300)
    xi, yi = np.meshgrid(xi, yi)
    plt.figure(figsize=(10,10))
    import scipy
    # Interpolate; there's also method='cubic' for 2-D data such as here
    for inter_method, num in zip(["nearest", "linear", "cubic"], [221, 222, 223]):
        plt.subplot(num)
        zi = scipy.interpolate.griddata((x1, x2), y, (xi, yi), method=inter_method)
        # sns.interactplot("x1", "x2", "y", data=df, levels=40)
        plt.imshow(zi, vmin=y.min(), vmax=y.max(), origin='lower',
                   extent=[x1.min(), x1.max(), x2.min(), x2.max()], cmap="rainbow")
        plt.title(inter_method)
        plt.colorbar()
    plt.legend()
    error = ""
    if plot_error:
        error = "error"
    plt.suptitle("Interpolated permability %s map" % (error))

    vis.plot_show("perm_map%s" % (error))


def pca_plot():
    data = WaterPermDataProvider()
    from sklearn.decomposition.pca import PCA
    pca = PCA(n_components=2)
    pcaX = pca.fit_transform(data.X)
    hist, bins = np.histogram(data.y, bins=8)
    y = np.digitize(data.y, bins=bins)
    y = bins[y-1]

    # plt.figure()
    df = pd.DataFrame(np.array((pcaX[:,0], pcaX[:,1], y)).transpose(), columns=["pc1", "pc2", "y"])
    sns.lmplot("pc1", "pc2", data=df, hue="y", fit_reg=False, palette="Reds")
    vis.plot_show("pca_plot")


def describe_data():
    data = WaterPermDataProvider()
    print(data.X.shape)


plot_weights(delta=150, kmin=5, kmax=25)

    # describe_data()
# pca_plot()
# plot_deltas(dmax=21)
# location_error_plot()
# location_error_plot(kind="other")
# permability_plot(plot_error=True)
# permability_plot()
# permability_plot("cubic")
# permability_plot("linear")
# permability_plot("nearest", plot_error=True)
# permability_plot("linear", plot_error=True)