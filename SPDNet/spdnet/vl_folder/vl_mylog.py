import numpy as np
from ..utils.diagLog import diagLog
from ..utils.diagInv import diagInv
from ..utils.symmetric import symmetric
from ..utils.dDiag import dDiag
from ..utils.my_svd import my_svd


from scipy import io

import os


def vl_mylog(X, dzdy=None):
    
    """LogEig Layer"""
    
    #print('X', X.shape)
    
    #if dzdy is not None:
    #    print('dzdy', dzdy.shape)
    
    
    EPS = 1e-12
    
    Us = np.zeros((X.shape[0], X.shape[1], X.shape[1]))
    Ss = np.zeros((X.shape[0], X.shape[1], X.shape[2]))
    Vs = np.zeros((X.shape[0], X.shape[2], X.shape[2]))
    
    
    for ix in range(0, X.shape[0]):
        Us[ix], Ss[ix], Vs[ix] = my_svd(X[ix])
    
    
    #print("Us", Us)
    
    #np.save(os.getcwd() + '/Us.npy', Us)
    #np.save(os.getcwd() + '/Vs.npy', Vs)
    #np.save(os.getcwd() + '/Ss.npy', Ss)
    
    #io.savemat(os.getcwd() + '/Us.mat', {'Us_my': Us})
    #io.savemat(os.getcwd() + '/Ss.mat', {'Ss_my': Ss})
    #io.savemat(os.getcwd() + '/Vs.mat', {'Vs_my': Vs})
    
    #print('U_transpose', Us.transpose(1, 2, 0).transpose(0, 2, 1))
    
    D = Ss[0].shape[1]
    Y = np.zeros((X.shape[0], X.shape[1], X.shape[1]))
    
    
    if dzdy is None:
        
        for ix in range(0, X.shape[0]):
            Y[ix] = Us[ix].dot(np.diag(np.log(np.diag(Ss[ix])))).dot(Us[ix].T)
        
        return Y
    
    else:
        
        for ix in range(0, X.shape[0]):
            
            U = Us[ix]
            S = Ss[ix]
            V = Vs[ix]
            
            diagS = np.diag(S).copy() # диагональ матрицы S
            
            ind = diagS > (D * EPS) # Здесь полная херь
            
            #print('ind', ind)
            
            Dmin = np.minimum(np.flatnonzero(ind.T)[-1] , D)
            
            #print('Dmin', Dmin)
            
            S = S[:, ind]
            U = U[:, ind]
            
            #print("dasdsa")
            dLdC = symmetric(dzdy[ix, :].reshape(D, D, order = 'F'))
            
            #print('dLdC', dLdC)
            
            
            #print('diag_log', diagLog(S, 0))
            
            #print('U', U[-10:, -10:])
            
            dLdV = 2 * dLdC.dot(U).dot(diagLog(S, 0))
            
            #print('dLdV', dLdV)
            
            dLdS = diagInv(S).dot(U.T.dot(dLdC).dot(U))
            
            #print(dLdS)
            
            if np.sum(ind) == 1:
                #K = 1. / (S[1].dot(np.ones((1, Dmin))) - (S[1].dot(np.ones((1, Dmin)))).T)
                #K[np.eye(K.shape[0]) > 0] = 0
                K = np.diag(S).copy().reshape(-1,)
                K = (np.repeat(K.reshape(1,-1), K.shape[0], axis=0) - np.repeat(K.reshape(-1,1), K.shape[0], axis=1)).T
                K = np.where(np.abs(K) < 1e-6, np.inf, K)
                K = 1. / K
            
            else:
                #K = 1./(np.diag(S).dot(np.ones((1, Dmin))) - (np.diag(S).dot(np.ones((1, Dmin)))).T)
                #K[np.eye(K.shape[0]) > 0] = 0
                #isinf = np.where(np.isinf(K), 1, K)
                #k[np.flatnonzero(isinf.T)] = 0
                K = np.diag(S).copy().reshape(-1,)
                K = (np.repeat(K.reshape(1,-1), K.shape[0], axis=0) - np.repeat(K.reshape(-1,1), K.shape[0], axis=1)).T
                K = np.where(np.abs(K) < 1e-6, np.inf, K)
                K = 1. / K
            if np.all(diagS == 1):
                dzdx = np.zeros((D, D))
            
            else:
                dzdx = U.dot(symmetric(K.T * (U.T.dot(dLdV))) + dDiag(dLdS)).dot(U.T)
            
            Y[ix] = dzdx
            
        #print(Y)
        
        return Y
