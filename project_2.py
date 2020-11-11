# w6
from __future__ import division
import numpy as np
import sys

X_train = np.genfromtxt(sys.argv[1], delimiter=",")
y_train = np.genfromtxt(sys.argv[2])
X_test = np.genfromtxt(sys.argv[3], delimiter=",")

#https://github.com/emedinac/CSMM.102x_Machine_Learning__edX/blob/master/Project2/hw2_classification.py

## can make more functions if required
def Countclasses(y_train): # just a counter per class
    Prior = []
    total = len(y_train)
    K_classes = np.unique(y_train)
    for i in K_classes:
        Prior.append(np.uint8(y_train==i).sum()/total)
    return Prior

def Probability(x, u, D): # Gaussian Distribution for MLE
    exponential_term = np.exp(-0.5 *    (np.matmul((x-u) , np.linalg.pinv(D))  * (x-u)).sum(-1)    )
    return ( exponential_term / np.sqrt(np.linalg.det(D)) ).squeeze() 

def ClassConditionalDensity(X_train, y_train): # 
    K_classes = np.unique(y_train)
    mean_y = []
    cov_y = []
    for i in K_classes:
        mask = y_train==i
        mean_y.append(  X_train[mask].sum(0)/len(X_train[mask])  )
        cov_y.append(    np.matmul(  (X_train[mask]-mean_y[-1]).T , (X_train[mask]-mean_y[-1]) )/len(X_train[mask]   ) )

    return mean_y, cov_y

## can make more functions if required
def pluginClassifier(X_train, y_train, X_test):   
    # this function returns the required output
    Prior = Countclasses(y_train) # Prior Distribution
    mean_y, cov_y = ClassConditionalDensity(X_train, y_train) # u and Cov parameters
    Likelihood = np.zeros([X_test.shape[0], len(Prior)])
    for k in range(len(Prior)):
        Likelihood[:,k] =  Prior[k] * Probability(X_test, mean_y[k], cov_y[k]) # computing the Likelihood for Bayes Classifier
    Prob = Likelihood/Likelihood.sum(1)[:,None]
    return Prob
