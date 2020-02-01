import numpy as np
from utils import Gradient_Descent
from utils import sigmoid

class LR:
    def __init__(self):
        self.xtrain = None
        self.xtest = None
        self.ytrain = None
        self.ytest = None

    def fit(self, xtrain, ytrain, w,b, alpha, iterations):
        for i in range(100):
            w,b = Gradient_Descent(xtrain, ytrain, w,b, alpha)
        return w,b
    def pred(self,xtest,w,b):
        n=xtest.shape[0]
        m=xtest.shape[1]
        ypred=np.zeros(m)
        A=sigmoid(np.dot(w.T,xtest)+b)
        for i in range(ypred.size):
            if(A[0][i]>0.5):
                ypred[i]=1
            else:
                ypred[i]=0
        return ypred

    def predict_proba(self,xtest,w,b):
        A = sigmoid(np.dot(w.T, xtest) + b)
        return A