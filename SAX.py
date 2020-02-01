import numpy as np
from utils import normalisation_1d
from utils import PAA
from utils import Find_Break_points
from utils import alphabet_array
from scipy import spatial
import math
from sklearn.metrics import accuracy_score

class SAX:
    def __init__(self, p, alpha):
        self.p = p
        self.alpha = alpha
        self.x = None

    def fit(self, xtrain,ytrain):
        n=ytrain.size
        alphabet=alphabet_array(self.alpha)
        alphabet=np.flip(alphabet)
        break_points=Find_Break_points(self.alpha)
        break_points=np.delete(break_points,0)
        sax_array=np.empty((n,1),dtype="str")
        temp=[]
        for k in range(n):
            paa_array = PAA(xtrain[k], self.p)
            temp.clear()
            for i in range(paa_array.size):
                test = 1
                for j in range(break_points.size):
                    if paa_array[i] <= break_points[j]:
                        temp += alphabet[j]
                        test = 0
                        break
                if test:
                    temp += 'a'
            str=''.join(temp)
            sax_array[k][0]=str
        words_set=set()
        classes_set=set()
        temp=[]
        for i in range(n):
            words_set.add(sax_array[i][0])
            classes_set.add(ytrain[i])
        for i in words_set:
            temp.append(i)
        words=np.asarray(temp)
        temp.clear()
        for i in classes_set:
            temp.append(i)
        classes=np.asarray(temp,dtype=int)
        num_words_per_class=np.zeros(classes.size)
        for i in range(classes.size):
            for j in range(np.size(sax_array,0)):
                if(ytrain[j]==classes[i]):
                    num_words_per_class[i]+=1
        ftd = np.zeros((words.size, classes.size))
        for i in range(words.size):
            for j in range(classes.size):
                for k in range(ytrain.size):
                    if(words[i]==sax_array[k] and ytrain[k]==classes[j]):
                        ftd[i][j]+=1
        ftd/=num_words_per_class
        ftd=np.log(1+ftd)
        num_of_classes_a_word_appear=np.zeros(words.size)
        class_test=np.ones((words.size,classes.size))
        k=0
        for i in range(words.size):
            for j in range(ytrain.size):
                for k in range(classes.size):
                    if(ytrain[j]==classes[k]): break
                if(words[i]==sax_array[j] and class_test[i][k]):
                    class_test[i][k]=0
                    num_of_classes_a_word_appear[i]+=1
        idf=np.log(classes.size/num_of_classes_a_word_appear)
        print(ftd.shape)
        print(idf.shape)
        VSM_train=np.zeros(ftd.shape)
        for i in range(ftd.shape[0]):
            VSM_train[i]=ftd[i]*idf[i]
        return VSM_train,classes


    def pred(self,xtest,VSM_train,ytest,classes):
        n = ytest.size
        alphabet = alphabet_array(self.alpha)
        alphabet = np.flip(alphabet)
        break_points = Find_Break_points(self.alpha)
        break_points = np.delete(break_points, 0)
        sax_array = np.empty((n, 1), dtype="str")
        temp = []
        for k in range(n):
            paa_array = PAA(xtest[k], self.p)
            temp.clear()
            for i in range(paa_array.size):
                test = 1
                for j in range(break_points.size):
                    if paa_array[i] <= break_points[j]:
                        temp += alphabet[j]
                        test = 0
                        break
                if test:
                    temp += 'a'
            str=''.join(temp)
            sax_array[k][0] = str
        words_set = set()
        temp = []
        for i in range(n):
            words_set.add(sax_array[i][0])
        for i in words_set:
            temp.append(i)
        words = np.asarray(temp, dtype="str")
        VSM_unlabeled=np.zeros(words.size)
        for i in range(words.size):
            for j in range(n):
                if(words[i]==sax_array[j]):
                    VSM_unlabeled[i]+=1
        VSM_unlabeled/=n
        VSM_train=np.transpose(VSM_train)
        max=-math.inf
        indice=0
        for i in range(np.size(VSM_train,0)):
            a=np.sqrt(np.sum(VSM_unlabeled**2))*np.sqrt(np.sum(VSM_train[i]**2))
            if(a>max):
                max=a
                indice=i
        ypred=np.zeros(ytest.size)
        label=classes[indice]
        for i in range(n):
            ypred[i]=label
        return accuracy_score(ytest,ypred)

    def SAX_representation(self, alpha, x):
        alphabet = alphabet_array(alpha)
        alphabet=np.flip(alphabet)
        temp = []
        break_points = Find_Break_points(alpha)
        break_points=np.delete(break_points,0)

        print("2-breaking points: ",break_points)
        print()
        paa_array = PAA(x, self.p)
        print("3-PAA array: ",paa_array)
        print()
        test = 1
        j = 0
        for i in range(paa_array.size):
            test = 1
            for j in range(break_points.size):
                if paa_array[i] <= break_points[j]:
                    temp += alphabet[j]
                    test = 0
                    break
            if test:
                temp += 'a'
        sax_array = np.asarray(temp)
        return sax_array
