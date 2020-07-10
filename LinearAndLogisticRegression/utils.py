import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib import pyplot as plt

def normal_equation(Xe, y):
    beta = np.linalg.inv(Xe.T.dot(Xe)).dot(Xe.T).dot(y)
    return beta

def extend_matrix(Xe):
    return np.c_[np.ones((len(Xe))), Xe]

def cost_function(Xe, y, beta):
    j = np.dot(Xe, beta) - y
    J = (j.T.dot(j)) / len(Xe) 
    return J

def normalize_values(Xn):
    Xn_mean = Xn.mean()
    Xn_std = np.std(Xn, ddof=1)
    return (Xn - Xn_mean) / Xn_std

def normalize_matrix(X):
    std = np.std(X, axis=0, ddof=1)
    mean = np.mean(X, axis=0)
    Xn = np.zeros((np.size(X, axis=0),np.size(X, axis=1)))
    for i in range(0, len(X[0,:])):
        col_row = (np.subtract(X[:,i], mean[i])) / std[i]
        Xn[:,i] = col_row
    return Xn

def normalize_matrix_with_value(X, x):
    std = np.std(X, axis=0, ddof=1)
    mean = np.mean(X, axis=0)
    xn = np.zeros(x.shape)
    for i in range(0, len(x[0])):
        col = (np.subtract(x[0][i], mean[i])) / std[i]
        xn[0][i] = col
    return xn

def normalize_vector(X, x):
    std = np.std(X, axis=0, ddof=1)
    mean = np.mean(X, axis=0)
    xn = np.zeros(x.shape)
    for i in range(0, len(x)):
        col = (np.subtract(x[i], mean[i])) / std[i]
        xn[i] = col
    return xn

def cost_function(Xe, y, beta):
    j = np.dot(Xe, beta) - y
    J = (j.T.dot(j)) / len(Xe) 
    return J

def gradient_decent(Xe, y, n_iterations, learning_rate):
    beta = np.array([0] * len(Xe[0]))
    for iteration in range(0, n_iterations):
        beta = np.subtract(beta, np.dot(np.dot(learning_rate, Xe.T), np.subtract(np.dot(Xe, beta), y)))
    return beta

def gradient_decent_2(Xe, y, n_iterations, learning_rate):
    beta_start = np.array([0] * np.size(Xe, 1))
    for iteration in range(0, n_iterations):
        beta_start = np.subtract(beta_start, np.dot((learning_rate / np.size(Xe, 0)), (np.dot(Xe.T, (sigmoid(np.dot(Xe, beta_start)) - y)))))
    return beta_start

def sigmoid(X):
    return np.divide(1, (np.add(1, np.e**(-X))))

def logistic_cost(beta, y):
    return np.dot((np.divide(-1, np.size(beta))),np.add(np.dot(y.T, np.log(sigmoid(beta))), np.dot((1 - y).T, np.log(1 - sigmoid(beta)))))

def mapFeatures(X1, X2, degree):
    X1_X2 = np.c_[X1,X2]
    Xe = extend_matrix(X1_X2)
    for i in range(2, degree + 1):
        for j in range(0, i + 1):
            X_new = Xe[:,1]**(i - j) * Xe[:,2]**j
            X_new = X_new.reshape(-1,1)
            Xe = np.append(Xe, X_new, 1)
    return Xe

def erros_and_accuracy(X, y, train_betas):
    p = sigmoid(np.dot(X, train_betas))
    pp = np.round(p)
    errors_train = np.sum(y!=pp)
    print('Traning errors:',errors_train )
    print('Training accuracy: %0.2f%%' % (100 - round(errors_train/len(y),3) * 100)) 

def plot_cost(Xne, y, n_iterations, learning_rate, sub=False):
    if(sub):
        plt.figure(figsize=(12,6))
        plt.subplot(121)
    x = np.arange(n_iterations)
    ys = np.zeros(n_iterations)
    for i in range(0, n_iterations):
        betas = gradient_decent_2(Xne,y, i, learning_rate)
        cost = logistic_cost(Xne.dot(betas), y)
        ys[i] = cost

    plt.plot(x,ys)
    plt.ylabel('J(Beta)')
    plt.xlabel('Number of iterations')
    print('Learning rate:',learning_rate)
    print('Number of iterations:',n_iterations)
    print('Cost:', round(cost, 4))

def plotting_boundaries(Xn, betas, y, degree):
    h = 0.01
    x_min, x_max = Xn[:,0].min()-0.1, Xn[:,0].max()+0.1
    y_min, y_max = Xn[:,1].min()-0.1, Xn[:,1].max()+0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    x1, x2 = xx.ravel(), yy.ravel()
    XXe = mapFeatures(x1, x2, degree)
    probabilites = sigmoid(np.dot(XXe, betas))
    classes = probabilites > 0.5
    mesh_classes = classes.reshape(xx.shape)
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF']) # mesh plot
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF']) # colors
    
    plt.subplot(122)
    plt.pcolormesh(xx, yy,mesh_classes, cmap=cmap_light)
    plt.scatter(Xn[:,0], Xn[:,1], c=y, marker='.', cmap=cmap_bold)