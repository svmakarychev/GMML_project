import numpy as np

def my_svd(M):
    
    """
        This function calculates svd decomposition of the matrix M in the matlab form
    """
    
    k = np.min(M.shape)
    S = np.zeros(M.shape)
    
    U, s, Vt = np.linalg.svd(M, full_matrices=True)
    S[:k, :k] = np.diag(s)
    
    return (U, S, Vt.T)
