import numpy as np

def vl_mybfc(X, W, dzdy):
    
    """
        BiMap Layer
        """
    
    Y = [0] * np.max(X.shape)
    
    for ix in range(0, np.max(X.shape)):
        Y[ix] = W.T.dot(X[ix]).dot(W)
    
    if dzdy is not None:
        
        rows, cols = W.shape
        Y_w = np.zeros((rows, cols))
        
        for ix in range(0, X.shape[0]):
            
            if type(dzdy) == np.ndarray:
                d_t = dzdy[ix]
            
            else:
                d_t = dzdy[:, ix]
                d_t = d_t.reshape((cols, cols))
        
        Y[ix] = W.dot(d_t).dot(W.T)
        Y_w = Y_w + 2 * X[ix] * W * d_t
        
        return Y, Y_w
    
    
    else:
        return Y
