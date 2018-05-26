import numpy as np

def vl_mylog(X, dzdy):
    
    """LogEig Layer"""
    
    EPS = 1e-12
    
    Us = [0] * np.max(X.shape)
    Ss = [0] * np.max(X.shape)
    Vs = [0] * np.max(X.shape)
    
    
    for ix in range(0, np.max(X.shape)):
        
        # Check this moment!
        # There are some difference between svd in Matlab and Python
        U, s, Vt = np.linalg.svd(X[ix], full_matrices=True)
        S = np.zeros((U.shape[1], Vt.shape[0]))
        S[:s.shape[0], :s.shape[0]] = np.diag(s)
        
        Us[ix] = U
        Ss[ix] = S
        Vs[ix] = Vt.T
    
    D = Ss[0].shape[1]
    Y = [0] * np.max(X.shape)
    
    if dzdy is None:
        
        for ix in range(0, np.max(X.shape)):
            Y[ix] = Us[ix].dot(np.diag(np.log(np.diag(Ss[ix])))).dot(Us[ix].T)
        
        return Y

    else:
        
        for ix in range(0, np.max(X.shape)):
            
            U = Us[ix]
            S = Ss[ix]
            V = Vs[ix]
            
            diagS = np.diag(S)
            ind = diagS > (D * EPS) # Здесь полная херь
            
            Dmin = np.minimum(np.flatnonzero(ind.T)[-1] , D)
            
            S = S[:, ind]
            U = U[:, ind]
            
            dldC = dzdy[:, ix].reshape((D, D))
            
            dldC = 1./2 * (dldC + dldC.T) # Check this! In his code something wrong
            
            dldV = 2 * dldC.dot(U).diaglog(S, 0)
            dldS = diagInv(S).dot(U.T.dot(dldC).dot(U))
            
            if np.sum(ind) == 1:
                K = 1. / (S[1].dot(np.ones((1, Dmin))) - (S[1].dot(np.ones((1, Dmin)))).T)
                K[np.eye(K.shape[0]) > 0] = 0
            
            else:
                K = 1./(np.diag(S).dot(np.ones((1, Dmin))) - (np.diag(S).dot(np.ones((1, Dmin)))).T)
                K[np.eye(K.shape[0]) > 0] = 0
                
                isinf = np.where(np.isinf(K), 1, K)
                k[np.flatnonzero(isinf.T)] = 0
            
            if np.all(diagS == 1):
                dzdx = np.zeros((D, D))
            
            else:
                nonsymmetric = K.T * (U.T.dot(dldV))
                symmetric = 1./ 2 * (nonsymmetric + nonsymmetric.T)
                
                dzdx = U.dot(symmetric + dDiag(dLdS)).dot(U.T)

return Y
