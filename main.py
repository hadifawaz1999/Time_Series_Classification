from sklearn.metrics import accuracy_score
from NN import NN
from SAX import SAX
from Logistic_Regression import LR
import matplotlib.pyplot as plt
import numpy as np
from utils import load_data
from utils import normalisation
from utils import normalisation_1d
from utils import load_data_images
from utils import normalisation_transpose
from tslearn.piecewise import OneD_SymbolicAggregateApproximation
import variables
import variables_image_recognition
import lime
import lime.lime_tabular
import sklearn
import sklearn.metrics
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import fetch_20newsgroups
from lime.lime_text import LimeTextExplainer
from utils_lime import Draw_Lime_On_Logistic_Regression

exit()

if variables.choice_of_work == 1:
    xtrain, ytrain, xtest, ytest = load_data(variables.file_name)
    xtrain, xtest = normalisation(xtrain, xtest)
    n = NN(variables.choice, variables.w)
    n.fit(xtrain, ytrain)
    ypred = n.pred(xtest)
    print(1 - accuracy_score(ytest, ypred))
    explainer = lime.lime_tabular.LimeTabularExplainer(xtrain, mode='classification')
    exp = explainer.explain_instance(xtest[0], n.pred_proba, labels=1, top_labels=1)
    exp.as_pyplot_figure()
    plt.show()
elif variables.choice_of_work == 2:
    xtrain, ytrain, xtest, ytest=load_data(variables.file_name)
    xtrain, xtest=normalisation(xtrain,xtest)
    sax=SAX(variables.p,variables.alpha)
    VSM_train, classes= sax.fit(xtrain,ytrain)
    score=sax.pred(xtest,VSM_train,ytest,classes)
    print(score)

elif variables.choice_of_work == 3:
    xtrain, ytrain, xtest, ytest = load_data_images(variables_image_recognition.file_name)
    xtrain, xtest = normalisation_transpose(xtrain, xtest)
    lr = LR()
    variables_image_recognition.w, variables_image_recognition.b = lr.fit(xtrain, ytrain, variables_image_recognition.w,
                                                                          variables_image_recognition.b,
                                                                          variables_image_recognition.alpha,
                                                                          variables_image_recognition.iterations)
    ypred = lr.pred(xtest, variables_image_recognition.w, variables_image_recognition.b)
    print("aaa", accuracy_score(ytest, ypred))
elif variables.choice_of_work == 4:
    xtrain, ytrain, xtest, ytest = load_data(variables.file)
    xtrain, xtest = normalisation(xtrain, xtest)
    Draw_Lime_On_Logistic_Regression(xtrain,ytrain,xtest,ytest)