import numpy as np
cimport numpy as np
from libc.float cimport DBL_MAX
import math
np.import_array()

def dtw1(xtrain, xtest, w):
    cdef int n,m,i,j,itrain,jtest,jstart,jend,condition_indice_verticly
    n=xtrain.size
    m=xtest.size
    cdef np.ndarray dtw = np.zeros((n+1,m+1), dtype=np.float64)
    dtw[:, 0] = DBL_MAX
    dtw[0, :] = DBL_MAX
    dtw[0][0] = 0
    for i in range(1, n + 1):
        jstart=max(1,i-w)
        jend=min(m+1,i+w+1)
        condition_indice_verticly = i - w - 1
        if condition_indice_verticly >= 0:
            dtw[i][condition_indice_verticly] = DBL_MAX
        for j in range(jstart, jend):
            itrain=i-1
            jtest=j-1
            dtw[i][j] = abs(xtrain[itrain] - xtest[jtest]) ** 2 + min(dtw[itrain][j],min(dtw[i][jtest],dtw[itrain][jtest]))
        if(jend<(m+1)):
            dtw[i][jend]=DBL_MAX
    return math.sqrt(dtw[n][m])

def dtw2(xtrain, xtest, w):
    cdef int n,m,i,j,itrain,jtest
    n = xtrain.size
    m = xtest.size
    cdef np.ndarray dtw = np.zeros((n + 1, m + 1), dtype=np.float64)
    dtw[:, 0] = DBL_MAX
    dtw[0, :] = DBL_MAX
    dtw[0][0] = 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            itrain=i-1
            jtest=j-1
            dtw[i][j] = abs(xtrain[i - 1] - xtest[j - 1]) ** 2 + min(dtw[i - 1][j],
                                                                     dtw[i][j - 1],
                                                                     dtw[i - 1][j - 1])
    return math.sqrt(dtw[n][m])