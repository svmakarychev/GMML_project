import numpy as np

def max_eig(D, c):
    
    """
        Add docstring later
        """
    
    if c is None:
        c = 0
    
    M, N = D.shape
    
    L = np.zeros(D.shape)
    
    m = np.minimum(M, N)
    h1 = np.diag(D)
    
    L_I = h_1 < c
    h_1(L_I) = c
    
    L[0:m, 0:m] = np.diag(h_1)
    
    return L, L_I
