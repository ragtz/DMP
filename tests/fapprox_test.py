from DMP.src.fapprox import *
from DMP.src.dmp import *
import numpy as np
import matplotlib
import matplotlib.pylab as plt
from sklearn.metrics import mean_squared_error as mse

matplotlib.rcParams.update({'font.size': 42})

if __name__ == '__main__':
    '''
    x = np.arange(-3,3,0.1)
    y = np.sin(x)
    t = np.cumsum(np.array(len(x)*[0.1]))-0.1
    
    X = np.array([[x[i], y[i]] for i in range(len(x))])
    
    #fapprox = linearInterpFunctApprox
    fapprox = radialBasisFunctApprox(100)
    
    f = fapprox(t, X)
    
    plt.plot(t, x)
    plt.plot(t, y)
    plt.plot(t, f(t)[:,0])
    plt.plot(t, f(t)[:,1])
    
    plt.figure()
    plt.plot(x, y)
    plt.plot(f(t)[:,0], f(t)[:,1])
    
    plt.show()
    '''
    '''
    x = np.arange(-1, 1, 0.01)
    y = np.sin(x)
    
    B = np.arange(5,30,5)
    S = np.arange(0.001, 0.5, 0.001)
    legend = []
    
    for b in B:
        print "Testing B =", b
        rms = []
        for s in S:
            fapprox = radialBasisFunctApprox(b, uniformPhaseStdDev(s))
            f = fapprox(x, y)
            rms.append(mse(y, f(x)))
    
        plt.plot(b*S, rms)
        legend.append('B = ' + str(b))
    
    plt.title('Basis Width vs MSE')
    plt.xlabel('B*std_dev')
    plt.ylabel('MSE')
    plt.legend(legend)
    plt.show()
    '''
    '''
    t = np.arange(0,16,0.1)
    x = t
    y = np.sin(x)
    
    X = np.array([[x[i], y[i]] for i in range(len(x))])
    
    demos = [(t, X)] 
    
    K = 1500.0
    D = 2*np.sqrt(K)
    eps = 0.01
    
    B = np.arange(5, 105, 5)
    legend = []
    
    rms = []
    for b in B:
        fapprox = radialBasisFunctApprox(b)
        dmp = DMP(K, D, eps, fapprox)
        rms.append(dmp.learn(demos))
        
    legend.append('Uniform Phase')   
    plt.plot(B, rms)
    
    rms = []
    for b in B:
        fapprox = radialBasisFunctApprox(b, uniformTime(eps))
        dmp = DMP(K, D, eps, fapprox)
        rms.append(dmp.learn(demos))
        
    legend.append('Uniform Time')
    plt.plot(B, rms)
    
    plt.title('Number of Bases vs MSE')
    plt.xlabel('B')
    plt.ylabel('MSE')
    plt.legend(legend)
    plt.show()
    '''
    
    t = np.arange(0,16,0.1)
    x = t
    y = np.sin(x)
    
    X = np.array([[x[i], y[i]] for i in range(len(x))])
    
    demos = [(t, X)] 
    
    K = 1500.0
    D = 2*np.sqrt(K)
    eps = 0.01
    
    x_0 = np.array(X[0,:])
    v_0 = np.array([0,0])
    g = np.array(X[-1,:])
    tau = 16.0
    dt = 0.1
    
    B = np.arange(5, 45, 5)
    legend = []
    
    rms = []
    for b in B:
        fapprox = radialBasisFunctApprox(b)
        dmp = DMP(K, D, eps, fapprox)
        dmp.learn(demos)
        rms.append(mse(X, dmp.plan(x_0, v_0, g, tau, dt)[1]))
        
    legend.append('Uniform Phase')   
    plt.plot(B, rms)
    
    rms = []
    for b in B:
        fapprox = radialBasisFunctApprox(b, uniformTime(eps))
        dmp = DMP(K, D, eps, fapprox)
        dmp.learn(demos)
        rms.append(mse(X, dmp.plan(x_0, v_0, g, tau, dt)[1]))
        
    legend.append('Uniform Time')
    plt.plot(B, rms)
    
    plt.title('Number of Bases vs MSE')
    plt.xlabel('B')
    plt.ylabel('MSE')
    plt.legend(legend)
    plt.show()
    
