import numpy as np
import scipy

def vl_mysoftmaxloss(X,c,dzdy):
    output = np.subtract(X, X.max(axis=0, keepdims=True))
    logarithms = np.log(np.sum(np.exp(output), axis=0))
    
    if dzdy is None:
        output = output - logarithms.reshape(-1,1)
        return output
    else:
        dzdy = dzdy - np.exp(output) * (np.sum(dzdy, axis=0) / np.sum(np.exp(output), axis=0)).reshape(-1,1)
        return dzdy
