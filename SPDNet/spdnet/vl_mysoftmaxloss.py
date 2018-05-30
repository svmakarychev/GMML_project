import numpy as np
import scipy

def vl_mysoftmaxloss(X,c,dzdy=None):    
        
    mass = np.ones((c.shape[0], ))
    
    c_ = c - 1
    
    for ic in range(0, len(c)):
        c_[ic] = c_[ic] + (ic) * X.shape[1]
    
    Xmax = np.max(X, 1)[:, None]
    ex = np.exp(X - Xmax)
    
    if dzdy is None:
        
        t = Xmax + np.log(np.sum(ex, 1))[:, None] - X.ravel()[c_][:, None]
        
        Y = np.sum(t)
   
    else:
        
        Y = (ex / np.sum(ex, 1)[:, None]).ravel()
        Y[c_] = Y[c_] - 1
        
        Y = Y.reshape(X.shape)

    return Y
    

