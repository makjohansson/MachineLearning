from matplotlib import pyplot as plt
import numpy as np
import matplotlib as mpl
from matplotlib.colors import ListedColormap
from sklearn.svm import SVC
from joblib import dump, load
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF','#FFCE00']) # mesh plot
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF', '#000000']) # colors

def plot_sv_decis(clf, X, y, tree=False):
    '''Plot decision boundarys. matplotlib.pyplot's function show() must be added after function to view the plot'''
    if tree:
        h = 0.1
    if not tree:
        h = 0.01
    X_min, X_max = X[:,0].min()-0.1, X[:,0].max()+0.1
    Y_min, Y_max = X[:,1].min()-0.1, X[:,1].max()+0.1
    XX, YY = np.meshgrid(np.arange(X_min, X_max, h), np.arange(Y_min, Y_max, h))
    x1, x2 = XX.ravel(), YY.ravel()  
    XY = np.vstack([x1, x2]).T
    classes = clf.predict(XY).reshape(XX.shape) 
    plt.pcolormesh(XX, YY, classes, cmap=cmap_light)
    plt.scatter(X[:,0], X[:,1], c=y, s=3, cmap=cmap_bold)

def train_or_load_clf(X, y, label, yy):
    '''Train or load classifiers. Return a SVC classifier with C=10 and gamma=0.01'''
    try:
        return load('utils/classifiers/one_vs_all_class' + str(label) + '.joblib')
    except FileNotFoundError:
        print('Training classifier', label)
        yy = (y == label).astype(int)
        clf = SVC(kernel='rbf', C=10, gamma=0.01, probability=True)
        clf.fit(X, yy)
        dump(clf, 'utils/classifiers/one_vs_all_class' + str(label) + '.joblib')
        return load('utils/classifiers/one_vs_all_class' + str(label) + '.joblib')

def one_vs_all(X, y, labels):
    '''Return a list of predictions'''
    predictions = np.zeros(y.shape)
    classifiers = []
    yy = np.zeros(y.shape)
    for label in range(0, labels):
            clf = train_or_load_clf(X, y, label, yy)
            classifiers.append(clf)
    
    for i in range(0, np.size(y)):
        pred = X[i].reshape(1, -1)    
        scores = []
        for label in range(0, labels):
            score = classifiers[label].predict_proba(pred)
            scores.append(score)

        prob_arr = np.array(scores).reshape(labels,2)

        
        highest_prob = prob_arr[:,1]
        highet_prob_list = highest_prob.tolist()
        prediction = highet_prob_list.index(max(highet_prob_list))
        predictions[i] = prediction

    return predictions

def restrict_cases(X, y):
    '''Restrict to cases where X[:,38] == 24'''
    h = np.where(X[:,38] == 24)
    X_24 = np.r_[X[h[0]]]
    y_24 = np.r_[y[h[0]]]
    return X_24, y_24

def tuning_max_depth(XX, yy, XX_val, yy_val, pick_tree='df', trees=3):
    train_results = []
    val_results = []
    max_depths =np.linspace(1, 10, num=10, endpoint=True)
    for max_depth in max_depths:
        if pick_tree == 'df':
            tree = DecisionTreeRegressor(max_depth=max_depth)
        elif pick_tree == 'rnd':
            tree = RandomForestRegressor(max_depth=max_depth, n_estimators=trees, random_state=10)
        tree.fit(XX, yy)
        pred_train = tree.predict(XX) # predict on training set
        mean_train = mean_squared_error(yy, pred_train) # mean square error on the traning set
        train_results.append(mean_train)

        pred_val = tree.predict(XX_val) # predictions on the validation set
        mean_val = mean_squared_error(yy_val, pred_val) # mean square error on the validation set
        val_results.append(mean_val)
    return val_results.index(min(val_results)) + 1, train_results, val_results

def plot_MNIST_fashion(X,y):
    plt.figure(1, figsize=(12,5))
    for i in range(16):
        plt.subplot(2,8,i+1)
        plt.imshow(X[i].reshape(28,28), cmap='gray',  interpolation='nearest')
        plt.title(y[i])
        plt.xticks([])
        plt.yticks([])
    plt.show()

def plot_loss_accuracy(history):
    plt.figure(1, figsize=(12, 6))
    plt.subplot(1,2,1)
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='valid')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='valid')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

