import numpy as np
import scipy

#надо понять, как все же подаются матрицы и выставить правильно размерности
def vl_myrec(X, epsilon, dzdy=None):
    svd_u = np.zeros((X.shape[2], X.shape[0], X.shape[0]))
    svd_s = np.zeros((X.shape[2], np.min(X.shape)))
    svd_v = np.zeros((X.shape[2], X.shape[1], X.shape[1]))
    
    for i in range(X.shape[0]):
        u, s, v = np.linalg.svd(X[i,:,:])
        svd_u[i,:,:] = u
        svd_s[i,:] = s
        svd_v[i,:,:] = v
    
    answer = X.copy()
    
    if dzdy is None:
        for i, data in enumerate(X):
            eigenvalues, eigenvectors = np.linalg.eig(data)
            eigenvalues = np.maximum(eigenvalues, epsilon)
            answer[i,:,:] = eigenvectors.dot(np.diag(eigenvalues).dot(eigenvectors.T))
        return answer
    
    else:
        for i in range(X.shape[0]):
            U, S, V = svd_u[i], np.diag(svd_s[i]), svd_v[i]
            
            Dmin = S.shape[0]
            
            dLdC = symmetric(dLdC)
            
            max_S, max_I = my_eigh(S, epsilon)
            dLdV = 2 * dLdC.dot(U).dot(max_S)
            dLdS = np.diag(np.where(dLdV == 0, dLdV))
            dLdS = dLdS.dot(U.T).dot(dLdC).dot(U)
            
            K = diag(S).reshape(-1,)
            K = np.repeat(K.reshape(1,-1), K.shape[0], axis=0) - np.repeat(K.reshape(-1,1), K.shape[0], axis=1)
            K = np.where(np.abs(K) < 1e-6, np.inf, K)
            K = 1. / K
            
            #dzdx = U*(symmetric(K'.*(U'*dLdV))+dDiag(dLdS))*U';
            #Y{ix} =  dzdx; %warning('no normalization');
            
            dzdx = U.dot(symmetric(K.T.dot(U.T.dot(dLdV)) + dDiag(dLdS))).dot(U.T)
            answer[i, :, :] = dzdx
        return answer
