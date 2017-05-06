import numpy as np
from sklearn.linear_model import LinearRegression

eps = 0.05

def uniformPhase(B, minX, maxX):
    std_dev = (maxX-minX)/float(B)
    means = np.arange(minX, maxX, std_dev)
    return (means, std_dev*np.ones(means.shape[0]))
    
def uniformPhaseStdDev(std_dev):
    def f(B, minX, maxX):
        s = (maxX-minX)/float(B)
        means = np.arange(minX, maxX, s)
        return (means, std_dev*np.ones(means.shape[0]))
    return f
    
def uniformTime(eps):
    def s(t):
        return np.exp(np.log(eps)*t)
        
    def spacing(B, minX, maxX):
        means = s(np.arange(0, 1, 1.0/B))
        std_devs = [means[1]-means[0]] + [means[i]-means[i-1] for i in range(1,means.shape[0])]
        return (means, std_devs)
        
    return spacing

def linearInterpFunctApprox(X, y):
    if X.shape[0] < 1:
        raise Exception("X must have more than 1 sample")
        
    if X.shape[0] != y.shape[0]:
        raise Exception("X and y must have the same number of samples")

    minX = np.min(X)
    maxX = np.max(X)
    
    idxs = np.argsort(X, kind='heapsort')
    X = X[idxs]
    y = y[idxs]
    
    def nearestXIdxs(x):
        if x < minX-eps or x > maxX+eps:
            raise Exception(str(x) + " not in range [" + str(minX) + "," + str(maxX) + "]")
            
        idx = np.argmin(np.abs(X-x))
        if idx == 0:
            return (0,1)
        elif idx == X.shape[0]-1:
            return (X.shape[0]-2,X.shape[0]-1)
        elif x >= X[idx] and x <= X[idx+1]:
            return (idx,idx+1)
        else:
            return (idx-1,idx)
        
    def interpolate(x, idxs):
        (i1, i2) = idxs
        return ((y[i2]-y[i1])/(X[i2]-X[i1]))*(x-X[i1]) + y[i1]
        
    return lambda X: np.array([interpolate(x, nearestXIdxs(x)) for x in X])

def radialBasisFunctApprox(B, spacing=uniformPhase):
    def fapprox(X, y):
        if X.shape[0] < 1:
            raise Exception("X must have more than 1 sample")
            
        if X.shape[0] != y.shape[0]:
            raise Exception("X and y must have the same number of samples")

        n_targets = 1 if len(y.shape) == 1 else y.shape[1]

        minX = np.min(X)
        maxX = np.max(X)
        
        (means, std_devs) = spacing(B, minX, maxX)
        
        coef = None
        intercept = None
        
        # return dim = (n_targets, n_features)
        def coef_intercept():
            axis = 0 if len(coef.shape) == 1 else 1
            return np.concatenate((coef, np.array([intercept]).T), axis=axis)
        
        def basis_vector(x):
            if x < minX-eps or x > maxX+eps:
                raise Exception(str(x) + " not in range [" + str(minX) + "," + str(maxX) + "]")
            return np.array([np.exp(-(x - means[b])**2/(2*std_devs[b]**2)) for b in range(B)] + [1.0])
        
        def basis_matrix(X):
            return np.array([basis_vector(x) for x in X])

        model = LinearRegression(fit_intercept=False)
        model.fit(basis_matrix(X), y)

        if n_targets > 1:
            coef = model.coef_[:,0:B]
            intercept = model.coef_[:,B].T
        else:
            coef = model.coef_[0:B]
            intercept = model.coef_[B]
            
        return lambda X: np.dot(np.array([basis_vector(x) for x in X]), coef_intercept().T)
    return fapprox

