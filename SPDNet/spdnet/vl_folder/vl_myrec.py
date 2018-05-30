import numpy as np
import scipy
from ..utils.my_svd import my_svd
from ..utils.max_eigh import max_eigh
from ..utils.symmetric import symmetric
from ..utils.dDiag import dDiag

def vl_myrec(X, epsilon, dzdy=None):
        
    svd_u = np.zeros((X.shape[0], X.shape[1], X.shape[1]))
    svd_s = np.zeros((X.shape[0], X.shape[1], X.shape[2]))
    svd_v = np.zeros((X.shape[0], X.shape[2], X.shape[2]))
    
    for i in range(X.shape[0]):
        svd_u[i], svd_s[i], svd_v[i] = my_svd(X[i])
        
        
    answer = X.copy()
    
    if dzdy is None:
        for i in range(X.shape[0]):
            
            max_S, _ = max_eigh(svd_s[i], epsilon)
            answer[i] = svd_u[i].dot(max_S).dot(svd_u[i].T)
            
        return answer
    
    else:
        for i in range(X.shape[0]):
            U, S, V = svd_u[i], svd_s[i], svd_v[i]
            
            Dmin = S.shape[0]
            
            dLdC = symmetric(dzdy[i])
            
            max_S, max_I = max_eigh(S, epsilon)
            dLdV = 2 * dLdC.dot(U).dot(max_S)
            
            dLdS = np.diag(np.where(max_I == 0, 1, 0)).copy()
            
            dLdS = dLdS.dot(U.T).dot(dLdC).dot(U)
            
            K = np.diag(S).copy().reshape(-1,)
            K = (np.repeat(K.reshape(1,-1), K.shape[0], axis=0) - np.repeat(K.reshape(-1,1), K.shape[0], axis=1)).T
            K = np.where(np.abs(K) < 1e-6, np.inf, K)
            K = 1. / K
            
            dzdx = U.dot(symmetric(K.T * (U.T.dot(dLdV))) + dDiag(dLdS)).dot(U.T)
            answer[i, :, :] = dzdx
        return answer
