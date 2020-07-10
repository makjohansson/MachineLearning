import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF']) # mesh plot
cmap_bold = ListedColormap(['#2100F2', '#FF0000']) # colors

############ Methods for k-NN Classification #################
def plot_OK_FAIL(X, y):
    markers = ['x', 'o']
    colors = ['red', '#09FF00']
    for i, c in enumerate(np.unique(y)):
        plt.scatter(X[:,0][y==c],X[:,1][y==c],c=colors[i], marker=markers[i], s=20)

def print_test_result(pred,test): 
    idx = 0
    for i in range(0, len(test)):
        idx = i + 1
        if(pred[i] == 0.):
            print('     Chip{}: [{}, {}] ==> Fail'.format(idx, test[i][0], test[i][1]))
        else:
            print('     Chip{}: [{}, {}] ==> OK'.format(idx, test[i][0], test[i][1]))

def get_classes(X, z, k=5):
    XX = np.copy(X[:, :2])
    distances = np.zeros(X.shape[0])
    for i in range(0, len(X)):
        distances[i] = np.linalg.norm(z - XX[i])
    X = np.c_[X, distances] 
    X = X[X[:,3].argsort()]
    neighbors = X[:k]
    most_freq_class = np.bincount(neighbors[:, 2].astype(int)).argmax()
    return most_freq_class

def kNN(X, z, k=5):
    predicitions = np.zeros(z.shape[0])
    for i in range(0, len(z)):
        predicitions[i] = get_classes(X, z[i], k)
    return predicitions

def plot_kNN(X, k, y):
    h=0.01
    X_min, X_max = X[:,0].min()-0.1, X[:,0].max()+0.1
    Y_min, Y_max = X[:,1].min()-0.1, X[:,1].max()+0.1
    XX, YY = np.meshgrid(np.arange(X_min, X_max, h), np.arange(Y_min, Y_max, h))
    x1, x2 = XX.ravel(), YY.ravel()  
    XY = np.vstack([x1, x2]).T
    classes = kNN(X, XY, k=k) 
    classes = np.atleast_2d(classes).reshape(XX.shape)
    plt.pcolormesh(XX, YY, classes, cmap=cmap_light)
    y = kNN(X, X[:, 2], k=k)
    plt.scatter(X[:,0], X[:,1], c=y, s=3, cmap=cmap_bold)

    #############################################################

    ############## Methods for k-NN Regression ##################

def kNN_regression(X, point, k=5):
    distances = np.zeros(X.shape[0])
    for i in range(0, len(X)):
        distances[i] = np.linalg.norm(point - X[i,0])
    X = np.c_[X, distances]
    X = X[X[:,2].argsort()]
    y = np.sum(X[:k,1]) / k
    return y

def get_y_values(X, points, k):
    y_values = np.zeros(points.shape[0])
    for point in range(0, len(points)):
        y_values[point] = kNN_regression(X, points[point], k=k)
    return y_values

def plot_kNN_regression(X, XX ):
    plt.scatter(X[:, 0], X[:,1], s=10)
    plt.plot(XX[:,0], XX[:,1], color='red')