import os
from scipy import io
import numpy as np
from pymanopt.manifolds import Stiefel

def accumulate_gradients(opts, lr, batchSize, net, res, mmap = None):
    
    for l in range(len(net['layers']) - 1, -1, -1):
        
        if res['dzdw'][l] is not None:
            
            if 'learningRate' not in net['layers'][l]:
                net['layers'][l]['learningRate'] = 1
            else:
                pass
            
            
            if 'weightDecay' not in net['layers'][l]:
                net['layers'][l]['weightDecay'] = 1
            else:
                pass
            
            
            thisLR = lr * net['layers'][l]['learningRate']
            
                
            if 'weight' in net['layers'][l]:
                
                
                if net['layers'][l]['type'] == 'bfc':
                    
                    W1 = net['layers'][l]['weight']
                    W1grad = (1. / batchSize) * res['dzdw'][l]
                    
                    manifold = Stiefel(W1.shape[0], W1.shape[1])
                    W1Rgrad = manifold.egrad2rgrad(W1, W1grad)
                    
                    net['layers'][l]['weight'] = manifold.retr(W1, -thisLR * W1Rgrad)
                
                else:
                    
                    net['layers'][l]['weight'] = net['layers'][l]['weight'] - thisLR * (1. / batchSize) * res['dzdw'][l]

            else:
                pass

    
    return net, res
