import numpy as np

def diagInv(X):
    """
        This function created diagonal matrix from inverced diagonal elements of the input matrix
        """
    
    diagX = np.diag(X)
    
    invX = np.diag(1. / diagX)
    
    return invX
