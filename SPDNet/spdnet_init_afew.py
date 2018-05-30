import numpy as np
from spdnet.utils.my_svd import my_svd

def spdnet_init_afew(seed=123):
    
    np.random.seed(seed)
    
    opts = {}
    
    opts['layernum'] = 3
    
    Winit = [0] * opts['layernum']
    
    opts['datadim'] = [400, 200, 100, 50]
    
    for iw in range(0, opts['layernum']):
        A = np.random.random((opts['datadim'][iw], opts['datadim'][iw]))
        
        U1, S1, V1 = my_svd(A.dot(A.T))
        
        Winit[iw] = U1[:, 0:opts['datadim'][iw + 1]]
    
    
    f = 1. / 100
    classNum = 7
    
    fdim = Winit[iw].shape[1] * Winit[iw].shape[1]
    
    theta = f * np.random.random((fdim, classNum))
    
    Winit.append(theta)
    
    net = {}
    
    net['layers'] = []
    
    net['layers'].append({'type': 'bfc', 'weight': Winit[0]})
    
    net['layers'].append({'type': 'rec'})
    
    net['layers'].append({'type': 'bfc', 'weight': Winit[1]})
    
    net['layers'].append({'type': 'rec'})
    
    net['layers'].append({'type': 'bfc', 'weight': Winit[2]})
    
    net['layers'].append({'type': 'log'})
    
    net['layers'].append({'type': 'fc', 'weight': Winit[-1]})
    
    net['layers'].append({'type': 'softmaxloss'})
    
    net['train_mode'] = True
    
    return net
