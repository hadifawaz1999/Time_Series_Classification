import numpy as np
from utils import load_data_images

file_name="Coffee"
xtrain,ytrain,xtest,ytest=load_data_images(file_name)
n=xtrain.shape[0]
m=xtrain.shape[1]

w=np.zeros((n,1))
b=0
alpha=0.0001
iterations=100