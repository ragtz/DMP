from fapprox import *
from scipy.ndimage.interpolation import shift

class DMP:
    def __init__(self, K, D, eps, fapprox):
        self.K = K
        self.D = D
        self.alpha = -np.log(eps)
        self.fapprox = fapprox
        self.f = None
        
    def s(self, tau, t):
        return np.exp(-self.alpha*t/tau)
        
    # demos = [(t0, x0), (t1,x1), (t2,x2)]
    def learn(self, demos):
        t = np.concatenate([ti for (ti,_) in demos]).astype(float)
        x = np.concatenate([xi for (_,xi) in demos])
        
        tau = float(np.mean([ti[-1]-ti[0] for (ti,_) in demos]))
        x_0 = np.mean([xi[0] for (_,xi) in demos], axis=0)
        g = np.mean([xi[-1] for (_,xi) in demos], axis=0)
    
        dt = np.array([shift(ti,1,order=0,cval=1.0) - shift(shift(ti,-1,order=0),1,order=0) for (ti,_) in demos])#.astype(float)
        if len(x.shape) == 1:
            dx = np.array([shift(xi,1,order=0) - shift(shift(xi,-1,order=0),1,order=0) for (_,xi) in demos])
            
            x_dot = dx/dt
            x_ddot = np.array([shift(xd,1,order=0) - shift(shift(xd,-2,order=0),2,order=0) for xd in x_dot])/dt
        else:
            dim = x.shape[1]
            dt = np.array([np.tile(d,(dim,1)).T for d in dt])
            dx = np.array([shift(xi,(1,0),order=0) - shift(shift(xi,(-1,0),order=0),(1,0),order=0) for (_,xi) in demos])
            
            x_dot = dx/dt
            x_ddot = np.array([shift(xd,(1,0),order=0) - shift(shift(xd,(-2,0),order=0),(2,0),order=0) for xd in x_dot])/dt
            
        x_dot = np.concatenate(x_dot)
        x_ddot = np.concatenate(x_ddot)
        
        v = tau*x_dot
        v_dot = tau*x_ddot
        
        s = self.s(tau, t)
        
        if len(x.shape) == 1:
            f_target = ((tau*v_dot + self.D*v)/self.K) - (g - x) + (g - x_0)*s
        else:
            dim = x.shape[1]
            f_target = ((tau*v_dot + self.D*v)/self.K) - (g - x) + (g - x_0)*np.tile(s,(dim,1)).T
        self.f = self.fapprox(s, f_target)
        
    def plan(self, x_0, v_0, g, tau, dt, obs=[], gamma=0.0, beta=0.0):
        T = np.arange(0, tau, dt)
        x = None
        v = None
        
        for i, t in enumerate(T):
            if i == 0:
                x = [x_0]
                v = [v_0]
            else:
                '''                
                if i == len(T)/4:
                    x[i-1] += np.array([0.0, 1.5])
                elif i == len(T)/2:
                    x[i-1] += np.array([0.5, 0.0])
                elif i == 3*len(T)/4:
                    x[i-1] += np.array([-0.5, 1.0])
                '''                
                s = self.s(tau, t)
                v_dot = ((self.K*(g - x[i-1]) - self.D*v[i-1] - self.K*(g - x[0])*s + self.K*self.f(np.array([s]))) / tau).flatten() + np.sum([gamma*np.exp(-beta*(x[i-1]-o)**2)*(x[i-1]-o)/np.linalg.norm(x[i-1]-o) for o in obs], axis=0)
                x_dot = v[i-1] / tau 
                
                x.append(x[i-1]+(x_dot*dt))
                v.append(v[i-1]+(v_dot*dt))

        return (T, np.array(x))
        
