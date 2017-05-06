from DMP.src.fapprox import *
from DMP.src.dmp import *
import numpy as np
import matplotlib
import matplotlib.pylab as plt

#matplotlib.rcParams.update({'font.size': 42})
matplotlib.rcParams.update({'font.size': 20})

def plotMotion(x, y, t, obs=None):
    plt.figure()
    plt.subplot(121)
    plt.plot(x, y, '.')
    plt.title('2-D Motion')
    plt.xlabel('X')
    plt.ylabel('Y')
    
    # plot obstacles
    if obs != None:
        plt.scatter([o[0] for o in obs], [o[1] for o in obs], s=500, c='r')
    
    plt.subplot(122)
    plt.plot(t, x, '.')
    plt.plot(t, y, '.')
    plt.title('Motion in Time')
    plt.xlabel('Time')
    plt.ylabel('X/Y')
    plt.legend(['X', 'Y'])
    
def plotPlannedMotion(dx, dy, dt, px, py, pt):
    plt.figure()
    plt.subplot(121)
    plt.plot(dx, dy, '.')
    plt.plot(px, py, '--', c='r')
    #plt.plot(px, py, '.')
    plt.title('2-D Motion')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend(['Demo', 'Plan'])
    #plt.legend(['Demo 1', 'Demo 2'])
    
    plt.subplot(122)
    plt.plot(dt, dx, '.')
    plt.plot(dt, dy, '.')
    plt.plot(pt, px, '--')
    plt.plot(pt, py, '--')
    #plt.plot(pt, px, '.')
    #plt.plot(pt, py, '.')
    plt.title('Motion in Time')
    plt.xlabel('Time')
    plt.ylabel('X/Y')
    plt.legend(['Demo X', 'Demo Y', 'Plan X', 'Plan Y'])
    #plt.legend(['Demo 1 X', 'Demo 1 Y', 'Demo 2 X', 'Demo 2 Y'])
    
def plotPlannedMotion2(dx1, dy1, dt1, dx2, dy2, dt2, px, py, pt):
    plt.figure()
    plt.subplot(121)
    plt.plot(dx1, dy1, '.')
    plt.plot(dx2, dy2, '.')
    plt.plot(px, py, '--')
    plt.title('2-D Motion')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend(['Demo 1', 'Demo 2', 'Plan'])
    
    plt.subplot(122)
    plt.plot(dt1, dx1, '.')
    plt.plot(dt1, dy1, '.')
    plt.plot(dt2, dx2, '.')
    plt.plot(dt2, dy2, '.')
    plt.plot(pt, px, '--')
    plt.plot(pt, py, '--')
    plt.title('Motion in Time')
    plt.xlabel('Time')
    plt.ylabel('X/Y')
    plt.legend(['Demo 1 X', 'Demo 1 Y', 'Demo 2 X', 'Demo 2 Y', 'Plan X', 'Plan Y'])

if __name__ == '__main__':
    t = np.arange(0,16,0.1)
    #t = np.arange(0,7,0.1)
    x = t
    #y = 10*np.sin(2*np.pi*0.3*x)
    y = np.sin(x)
    
    X = np.array([[x[i], y[i]] for i in range(len(x))])
    X2 = X + np.random.normal(0, 0.075, X.shape)
    
    theta = np.pi/8
    #theta = np.pi/2
    #T = [0.0, 2.0]
    T = [0.0, 0.0]
    R = np.matrix([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    Xp = np.array((R*X.T).T + T)
    '''
    t = np.array([0.0,0.1,0.3,0.4,0.7,1.0])
    X = np.array([[0.0,0.0],
                  [1.0,0.0],
                  [1.5,0.5],
                  [2.0,0.6],
                  [1.8,1.0],
                  [1.0,0.0]])
    '''
    demos = [(t, X)]#, (t, X2)] 
    
    K = 1500.0
    D = 2*np.sqrt(K)
    eps = 0.01
    
    #spacing = uniformPhase
    spacing = uniformTime(eps)
    
    fapprox = linearInterpFunctApprox
    #fapprox = radialBasisFunctApprox(15, spacing=spacing)
    
    dmp = DMP(K, D, eps, fapprox)
    dmp.learn(demos)

    x_0 = np.array(Xp[0,:])
    v_0 = np.array([0,0])
    g = np.array(Xp[-1,:])
    #g[0] *= 0.5
    tau = 16.0
    dt = 0.1
    obs = []#[[8.0, 0.5],[4.0,-1.0],[9.0,1.0]]
    gamma = 500.0
    beta = 10.0
    (new_t, new_X) = dmp.plan(x_0, v_0, g, tau, dt, obs, gamma, beta)
    
    #plotMotion(X[:,0], X[:,1], t, obs)
    #plotMotion(new_X[:,0], new_X[:,1], new_t, obs)
    
    plotPlannedMotion(X[:,0], X[:,1], t, new_X[:,0], new_X[:,1], new_t)
    #plotPlannedMotion(X[:,0], X[:,1], t, X2[:,0], X2[:,1], t)
    #plotPlannedMotion2(X[:,0], X[:,1], t, X2[:,0], X2[:,1], t, new_X[:,0], new_X[:,1], new_t)
    '''
    x_0 = np.array([0,0])
    v_0 = np.array([0,0])
    g = np.array([0.1,15.6])
    tau = 15.6
    dt = 0.1
    (new_t, new_X) = dmp.plan(x_0, v_0, g, tau, dt)
    
    new_x = [v[0] for v in new_X]
    new_y = [v[1] for v in new_X]
    
    plotMotion(new_x, new_y, new_t)
    '''
    plt.show()

