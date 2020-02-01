import numpy as np
import math
from utils import ed
# from utils import dtw1
# from utils import dtw2
from dtw_cython.dtw_functions_in_cython import dtw1
from dtw_cython.dtw_functions_in_cython import dtw2
from tslearn.metrics import dtw as dtw_tslearn
from utils import sigmoid


class NN:
    def __init__(self, choice, w):
        self.w = w
        if choice == 1:
            self.distance = ed
            self.xtrain = None
            self.ytrain = None
            self.xtest = None
            self.ypred = None
            self.proba = None
        elif choice == 2:
            self.distance = dtw2
            self.xtrain = None
            self.ytrain = None
            self.xtest = None
            self.ypred = None
        elif choice == 3:
            self.distance = dtw1
            self.xtrain = None
            self.ytrain = None
            self.xtest = None
            self.ypred = None

    def fit(self, xtrain, ytrain):
        self.xtrain = xtrain
        self.ytrain = ytrain

    def pred(self, xtest):
        self.proba=np.zeros(xtest.shape[0])
        indice = 0
        min = math.inf
        ypred = np.ones(xtest.shape[0])
        for i in range(xtest.shape[0]):
            min = math.inf
            for j in range(self.xtrain.shape[0]):
                distance = self.distance(self.xtrain[j], xtest[i], self.w)
                if (distance < min):
                    min = distance
                    indice = j
            self.proba[i]=min
            ypred[i] = self.ytrain[indice]
        return ypred

    def pred_proba(self,xtest):
        self.proba=sigmoid(self.proba)
        return self.proba