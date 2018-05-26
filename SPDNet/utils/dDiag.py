import numpy as np

def dDiag(M):
    """
        This function just multiply input matrix on identity matrix.
        But in the original it additionally transform input on GPU
        We dont use this ability because we don't have GPU for now
        """
    
    I = np.eye(M.shape[0], M.shape[1])
    
    output = I * M
    
    return output
