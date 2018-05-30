import numpy as np

def vl_myfunc(X, W, dzdy=None):
    
    
    X_t = X.reshape(X.shape[0], -1, order = 'F')
    
    if dzdy is None:
        
        return (X_t.dot(W))
    
    else:
        
        Y = dzdy.dot(W.T)
        Y_w = X_t.T.dot(dzdy)
        
        return Y, Y_w
