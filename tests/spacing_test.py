from DMP.src.fapprox import *
import numpy as np
import matplotlib
import matplotlib.pylab as plt

matplotlib.rcParams.update({'font.size': 42})

if __name__ == '__main__':
    eps = 0.01
    B = 15
    spacing = uniformPhase
    #spacing = uniformTime(eps)
    
    T = np.arange(0, 1, 0.001)
    
    (means, std_devs) = spacing(B, 0, 1)
    
    bases = np.array([[np.exp(-(t - means[b])**2/(2*std_devs[b]**2)) for t in T] for b in range(B)])
    
    plt.subplot(121)
    for b in range(B):
        plt.plot(T, bases[b])
    plt.title('Bases in s')
    plt.xlabel('s')
    plt.ylabel('f(s)')
        
    plt.subplot(122)
    for b in range(B):
        plt.plot(np.log(T)/np.log(eps), bases[b])
    plt.title('Bases in t')
    plt.xlabel('t')
    plt.ylabel('f(t)')
    
    plt.axis((0,1,0,1))
    plt.show()
