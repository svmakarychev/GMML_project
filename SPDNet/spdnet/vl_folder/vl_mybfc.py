import numpy as np

def vl_mybfc(X, W, dzdy=None):
    
    """
        BiMap Layer
    """
    
    Y = np.zeros((X.shape[0], W.shape[1], W.shape[1]))
    
    for ix in range(0, X.shape[0]):
        Y[ix] = W.T.dot(X[ix]).dot(W)
    
    if dzdy is not None:
        
        Y = np.zeros((X.shape[0], W.shape[0], W.shape[0]))
        
        rows, cols = W.shape
        Y_w = np.zeros((rows, cols))
        
        for ix in range(0, X.shape[0]):
            
            d_t = dzdy[ix]
                    
            Y[ix] = W.dot(d_t).dot(W.T)
            Y_w = Y_w + 2 * X[ix].dot(W).dot(d_t)
        
        return Y, Y_w
    
    
    else:
        return Y