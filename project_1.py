## Solution for Part 2
# score 13

# active learning
def part2():
    Xtest = X_test.copy()
    Xtrain = X_train.copy()
    indices = []
    sigma2 = Xtrain[:,-1].T.dot(Xtrain[:,-1])/sigma2_input + lambda_input * np.eye(Xtrain.shape[1]-1)
    for j in range(10):

        ind = 0
        val_max = 0
        for i, x in enumerate(Xtest):
            val = sigma2_input + x[:-1][np.newaxis].dot(np.linalg.inv(sigma2)).dot(x[:-1][np.newaxis].T)
            if val > val_max and (i+1) not in indices:
                val_max = val
                ind = i+1
        indices.append(int(ind))
        sigma2 = sigma2 + Xtest[ind-1,:-1][np.newaxis].T.dot(Xtest[ind-1,:-1][np.newaxis])
    return indices 
# might be right 
# https://github.com/hjk612/Columbia-Machine-Learning-Edx/blob/master/Project%203/hw1_regression.py
## Solution for Part 2
def part2(y_train,X_train, X_test):
    d = np.shape(X_train)[1]
    indexes = list(range(1,len(X_test)+1))
    sigma = np.linalg.inv(lambda_input*np.identity(d) + 
                          sigma2_input**(-1)*np.transpose(X_train)*X_train)
    loc = []
    while len(loc)<=10:
        VAR = []
        for i in range(len(X_test)):
            sigma_temp = np.linalg.inv(np.linalg.inv(sigma) + 
                                       sigma2_input**(-1)*np.transpose(X_test[i])*X_test[i])
            VAR.append(sigma2_input+X_test[i]*sigma_temp*np.transpose(X_test[i]))
        index = np.argmax(VAR)
        sigma = np.linalg.inv(np.linalg.inv(sigma) + 
                                       sigma2_input**(-1)*np.transpose(X_test[index])*X_test[index])
        X_test = np.delete(X_test,index,0)
        loc.append(indexes[index])
        indexes.remove(indexes[index])
    
    return loc
