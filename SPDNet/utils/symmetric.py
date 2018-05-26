import numpy as np

def symmetric(M):
    """
        This function symmetrized tensor.
        """
    
    output = 0.5 * (M + np.transpose(M, (1, 0, 2)))
    
    return output
