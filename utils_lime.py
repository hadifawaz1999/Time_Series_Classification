import numpy as np
import math
import sklearn
from sklearn.linear_model import LogisticRegression
from utils import load_data
from utils import normalisation
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt

def Draw_Lime_On_Logistic_Regression(xtrain,ytrain,xtest,ytest):
    lr = LogisticRegression()
    lr.fit(xtrain, ytrain)
    lr.predict(xtest)

    explainer = lime.lime_tabular.LimeTabularExplainer(xtrain, mode='regression')
    for i_size in range(ytest.size):
        exp = explainer.explain_instance(xtest[i_size], lr.predict, labels=(0, 1), top_labels=(0, 1))
        temp = exp.as_map()
        s = 0
        indice = 0
        max = -math.inf
        list=exp.as_list()
        print(list)
        exit()
        for j_labels in range(2):
            s = 0
            for j_lenght_temp in range(len(temp[j_labels])):
                s += temp[j_labels][j_lenght_temp][1]
            if (s > max):
                max = s
                indice = j_labels
        indices_list = []
        for i_lenght_temp_indice in range(len(temp[indice])):
            if temp[indice][i_lenght_temp_indice][1] >= 0:
                indices_list.append(temp[indice][i_lenght_temp_indice][0])
        indices = np.asarray(indices_list, dtype=int)
        indices = np.sort(indices)
        print(i_size, " : ", indices)
        subexplenation = []
        for i_lenght_indices in range(indices.size):
            if i_lenght_indices != indices.size - 1 and indices[i_lenght_indices] == indices[i_lenght_indices + 1] - 1:
                subexplenation.append(indices[i_lenght_indices])
            else:
                subexplenation.append(indices[i_lenght_indices])
                if (len(subexplenation) == 1):
                    plt.plot(subexplenation, xtest[i_size][subexplenation], color='green', marker='o',
                             markerfacecolor='green', markersize=6)
                else:
                    plt.plot(subexplenation, xtest[i_size][subexplenation], color='green', linewidth=6)
                print(subexplenation, "  :  ", xtest[i_size][subexplenation])
                subexplenation.clear()
        plt.plot(xtest[i_size], color='red')
        # exp.as_pyplot_figure()
    plt.show()
