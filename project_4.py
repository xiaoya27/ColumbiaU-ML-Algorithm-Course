# might right 
from __future__ import division
import numpy as np
import sys
import pandas as pd

data = pd.read_csv(sys.argv[1])
data = data.pivot(index = data.columns[0], columns = data.columns[1], 
                  values = data.columns[2])
M_mask = ~data.isnull().as_matrix()
train_data = data.as_matrix()

lam = 2
sigma2 = 0.1
d = 5
iterations = 50

# Implement function here
def PMF(train_data):
    iterations = 50
    L  = np.zeros(iterations)
    Nu = train_data.shape[0]
    Nv = train_data.shape[1]
    U  = np.zeros((iterations, Nu, d))
    V  = np.zeros((iterations, Nv, d))
    mean = np.zeros(d)
    cov  = (1/lam) * np.identity(d)
    V[0] = np.random.multivariate_normal(mean, cov, Nv)
    
    for itr in range(iterations):
        
        if itr == 0:
            l = 0
        else:
            l = itr - 1
        
        for i in range(Nu):
            Z1 = lam * sigma2 * np.identity(d)
            Z2 = np.zeros(d)
            for j in range(Nv):
                if train_data[i, j] == True:
                    Z1 += np.outer(V[l, j], V[l, j]) #movie rated by the user
                    Z2 += train_data[i, j] * V[l, j]    
            
            U[itr, i] = np.dot(np.linalg.inv(Z1), Z2)  
        for j in range(Nv):
            Z1 = lam * sigma2 * np.identity(d)
            Z2  = np.zeros(d)
            for i in range(Nu):
                if train_data[i, j] == True:
                    Z1 += np.outer(U[itr, i], U[itr, i])
                    Z2 += train_data[i, j] * U[itr, i]
            
            V[itr, j] = np.dot(np.linalg.inv(Z1), Z2)
            
        temp = 0 
        for i in range(Nu):
            
            for j in range(Nv):
                
                if train_data[i ,j] == True:
                    temp -= np.square(train_data[i ,j] - np.dot(U[itr, i].T, V[itr, j]))
                
                
            
        temp = (1/(2*sigma2)) * temp
        
        temp -= (lam/2) * (np.square(np.linalg.norm(U[itr])) + np.square(np.linalg.norm(V[itr])))
        
        L[itr] = temp 
    return L, U, V
# Assuming the PMF function returns Loss L, U_matrices and V_matrices (refer to lecture)
L, U_matrices, V_matrices = PMF(train_data, M_mask, lam, sigma2, d, iterations)

np.savetxt("objective.csv", L, delimiter=",")

np.savetxt("U-10.csv", U_matrices[9], delimiter=",")
np.savetxt("U-25.csv", U_matrices[24], delimiter=",")
np.savetxt("U-50.csv", U_matrices[49], delimiter=",")

np.savetxt("V-10.csv", V_matrices[9], delimiter=",")
np.savetxt("V-25.csv", V_matrices[24], delimiter=",")
np.savetxt("V-50.csv", V_matrices[49], delimiter=",")

# 15 scores

# w11
from __future__ import division
import numpy as np
import sys

train_data = np.genfromtxt(sys.argv[1], delimiter = ",")

lam = 2
sigma2 = 0.1
d = 5
def PMF(train_data):
    data_shape = (int(max(train_data[:, 0])) + 1, int(max(train_data[:, 1])) + 1)

    L, U_matrices, V_matrices = [], [], []

    V = np.random.standard_normal((d, data_shape[1])) / lam
    U = np.zeros((d,data_shape[0])) / lam

    for _ in range(50):
        for i, u in enumerate(U.T):
            sigma = np.zeros((d,d))
            Mv = np.zeros((d,1))
            for j in train_data[train_data[:,0]==i][:,1].astype(int):
                sigma += (V[:, j][np.newaxis].T.dot(V[:, j][np.newaxis])).copy()
                Ms = train_data[(train_data[:,0]==i) * (train_data[:,1]==j)][0,2]
                Mv += (V[:,j][np.newaxis].T*Ms).copy()
            ui = np.linalg.inv(lam*sigma2*np.eye(d)+sigma.copy()).dot(Mv.copy())
            U[:,i] = ui.ravel().copy()

        for i, v in enumerate(V.T):
            sigma = np.zeros((d,d))
            Mu = np.zeros((d,1))
            for j in train_data[train_data[:,1]==i][:,0].astype(int):
                sigma += U[:, j][np.newaxis].T.dot(U[:, j][np.newaxis])
                Mu += U[:,j][np.newaxis].T*(train_data[(train_data[:,0]==j) * (train_data[:,1]==i)][0,2])
            vi = np.linalg.inv(lam*sigma2*np.eye(d)+sigma).dot(Mu)
            V[:,i] = vi.ravel().copy()

        U_matrices.append(U.T.copy())
        V_matrices.append(V.T.copy())

        l = 0

        for j in range(train_data.shape[0]):
            res = U[:,int(train_data[j, 0])][np.newaxis].dot(V[:, int(train_data[j, 1])][np.newaxis].T)[0,0]
            l = l - 0.5 / sigma2 * (train_data[j, 2] - res) ** 2
        l = l - lam / 2 * np.sum(np.linalg.norm(V, 2, axis=0)) - lam / 2 * np.sum(np.linalg.norm(U, 2, axis=0))

        L.append(l)
        

    return L, U_matrices, V_matrices