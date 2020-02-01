import numpy as np
import os
import math
from scipy.stats import norm


def load_data(file_name):
    folder_path = "/home/hadi/data sets/UCRArchive_2018/"
    folder_path += (file_name + "/")
    train_path = folder_path + file_name + "_TRAIN.tsv"
    test_path = folder_path + file_name + "_TEST.tsv"
    if (os.path.exists(test_path) <= 0):
        print("File not found")
        return None, None, None, None
    train = np.loadtxt(train_path, dtype=np.float64)
    test = np.loadtxt(test_path, dtype=np.float64)
    ytrain = train[:, 0]
    ytest = test[:, 0]
    xtrain = np.delete(train, 0, axis=1)
    xtest = np.delete(test, 0, axis=1)
    return xtrain, ytrain, xtest, ytest


def load_data_images(file_name):
    folder_path = "/home/hadi/data sets/UCRArchive_2018/"
    folder_path += (file_name + "/")
    train_path = folder_path + file_name + "_TRAIN.tsv"
    test_path = folder_path + file_name + "_TEST.tsv"
    if (os.path.exists(test_path) <= 0):
        print("File not found")
        return None, None, None, None
    train = np.loadtxt(train_path, dtype=np.float64)
    test = np.loadtxt(test_path, dtype=np.float64)
    ytrain = train[:, 0]
    #ytrain.shape = (1, ytrain.size)
    ytest = test[:, 0]
    #ytest.shape = (1, ytest.size)
    xtrain = np.delete(train, 0, axis=1)
    xtrain = np.transpose(xtrain)
    xtest = np.delete(test, 0, axis=1)
    xtest = np.transpose(xtest)
    return xtrain, ytrain, xtest, ytest


def normalisation(xtrain, xtest):
    xtrain = (xtrain - xtrain.mean(axis=1, keepdims=True)) / (xtrain.std(axis=1, keepdims=True))
    xtest = (xtest - xtest.mean(axis=1, keepdims=True)) / (xtest.std(axis=1, keepdims=True))
    return xtrain, xtest


def normalisation_transpose(xtrain, xtest):
    xtrain = (xtrain - xtrain.mean(axis=0, keepdims=True)) / (xtrain.std(axis=0, keepdims=True))
    xtest = (xtest - xtest.mean(axis=0, keepdims=True)) / (xtest.std(axis=0, keepdims=True))
    return xtrain, xtest


def normalisation_1d(x):
    x = (x - np.average(x)) / (np.std(x))
    return x


def ed(xtrain, xtest, w):
    a = np.linalg.norm(xtrain - xtest)
    return a


def dtw1(xtrain, xtest, w):
    n = xtrain.size
    m = xtest.size
    r = w
    # r = int(w * max(n, m))
    dtw = np.zeros((n + 1, m + 1), dtype=np.float64)
    dtw[:, 0] = math.inf
    dtw[0, :] = math.inf
    dtw[0][0] = 0
    for i in range(1, n + 1):
        jstart = max(1, i - r)
        jend = min(m + 1, i + r + 1)
        condition_indice_verticly = i - r - 1
        if condition_indice_verticly >= 0:
            dtw[i][condition_indice_verticly] = math.inf
        for j in range(max(1, i - r), min(m + 1, i + r + 1)):
            dtw[i][j] = abs(xtrain[i - 1] - xtest[j - 1]) ** 2 + min(dtw[i - 1][j],
                                                                     dtw[i][j - 1],
                                                                     dtw[i - 1][j - 1])
        if (jend < (m + 1)):
            dtw[i][jend] = math.inf
    return math.sqrt(dtw[n][m])


def dtw2(xtrain, xtest, w):
    n = xtrain.size
    m = xtest.size
    dtw = np.zeros((n + 1, m + 1), dtype=np.float64)
    dtw[:, 0] = math.inf
    dtw[0, :] = math.inf
    dtw[0][0] = 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dtw[i][j] = abs(xtrain[i - 1] - xtest[j - 1]) ** 2 + min(dtw[i - 1][j],
                                                                     dtw[i][j - 1],
                                                                     dtw[i - 1][j - 1])
    return math.sqrt(dtw[n][m])


def Find_Break_points(alpha):
    break_points = np.arange(0, alpha, 1) / alpha
    return norm.ppf(break_points)


def alphabet_array(alpha):
    alphabet = np.empty(1, )
    for i in range(97, 97 + alpha):
        alphabet = np.append(alphabet, chr(i))
    alphabet = np.delete(alphabet, 0)
    return alphabet


def PAA(x, p):
    pp = x.size // p
    i, j = 0, p
    temp = []
    while True:
        indices = np.arange(i, j, 1)
        temp += x[indices].tolist()
        i, j = i + p, j + p
        if j > x.shape[0]:
            break
    paa_array = np.asarray(temp, dtype=np.float64)
    paa_array.shape = (p, pp)
    paa_array = paa_array.mean(axis=1, keepdims=False)
    rest = x.shape[0] % p
    if rest > 0:
        indices2 = np.arange(x.shape[0] - rest, x.shape[0], 1)
        temp_rest_array = x[indices2]
        res_point = temp_rest_array.mean()
        paa_array = np.append(paa_array, res_point)
    return paa_array


def sigmoid(z):
    x = 1 / (1 + np.exp(-z))
    return x


def Gradient_Descent(xtrain, ytrain, w,b, alpha):
    n = xtrain.shape[0]
    m = xtrain.shape[1]
    A = sigmoid(np.dot(w.T, xtrain)+b)
    dw = (1 / m) * np.dot(xtrain, (A - ytrain).T)-2*w
    db = (1 / m) * np.sum(A - ytrain)-2*b
    w = w - alpha * dw
    b = b - alpha * db
    return w,b