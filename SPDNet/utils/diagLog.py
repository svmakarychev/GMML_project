import numpy as np

def diagLog(D, c):
    """
        Computes log of a diagonal matrix
        Add constant displacement c if necessary
        """
    
    if c is None:
        c = 0
    
    M, N = D.shape
    
    L = np.zeros(D.shape)
    
    m = np.minimum(M, N)
    
    L[0:m, 0:m] = np.diag(np.log(np.diag(D) + c))
    
    return L
