import numpy as np
from .vl_mybfc import vl_mybfc
from .vl_myfunc import vl_myfunc
from .vl_myrec import vl_myrec
from .vl_mylog import vl_mylog
from .vl_mysoftmaxloss import vl_mysoftmaxloss


# I dont use doder, because I can't understand what arguments is not None
# and in the main function, this function is called from 4 arguments
def vl_myforbackward(net, x, res, dzdy=None, varargin=None):
    """
        Evalutes a simple SPDNet
        Forward and Backward passes
    """
    
    
    opts = {}
    opts['res'] = []
    opts['conserveMemory'] = False
    opts['sync'] = False
    opts['disableDropout'] = False
    opts['freezeDropout'] = False
    opts['accumulate'] = False
    opts['cudnn'] = True
    opts['skipForward'] = False
    opts['backPropDepth'] = + np.inf
    opts['epsilon'] = 1e-4
    
    
    n = len(net['layers'])
    
    res = {}
    
    res['x'] = [0] * (n+1)
    res['dzdx'] = [0] * (n+1)
    res['dzdw'] = [0] * (n+1)
    res['aux'] = [0] * (n+1)
    ###############################
    # This moment I don't understand exactly
    res['time'] = [0] * (n+1)
    res['backwardTime']  = [0] * (n+1)
    ######################
    
    if not opts['skipForward']:
        res['x'][0] = x

    ##########################
    # Forward pass
    #########################
    
    for i in range(0, n):
        
        if opts['skipForward']:
            break
        
        l = net['layers'][i]
        
        if l['type'] == 'bfc':
            res['x'][i+1] = vl_mybfc(res['x'][i], l['weight'])
        
        elif l['type'] == 'fc':
            res['x'][i+1] = vl_myfunc(res['x'][i], l['weight'])
        
        elif l['type'] == 'rec':
            res['x'][i+1] = vl_myrec(res['x'][i], opts['epsilon'])
        
        elif l['type'] == 'log':
            res['x'][i+1] = vl_mylog(res['x'][i])
        
        elif l['type'] == 'softmaxloss':
            res['x'][i+1] = vl_mysoftmaxloss(res['x'][i], l['class'])
        
        elif l['type'] == 'custom':
            res['x'][i+1] = l.forward(l, res)
        
        else:
            raise Exception("Unknown layer type")

        # I did not implement block
        # that can forget intermediate results

    
    if net['train_mode'] == False:
        return res
    
    ##########################
    # Backward pass
    #########################

    res['dzdx'][n] = dzdy

    for i in range(n-1, -1, -1):
    
        l = net['layers'][i]
    
        if l['type'] == 'bfc':
            res['dzdx'][i], res['dzdw'][i] = vl_mybfc(res['x'][i], l['weight'], res['dzdx'][i + 1])
        
        elif l['type'] == 'fc':
            res['dzdx'][i], res['dzdw'][i] = vl_myfunc(res['x'][i], l['weight'], res['dzdx'][i + 1])

        elif l['type'] == 'rec':
            res['dzdx'][i] = vl_myrec(res['x'][i], opts['epsilon'], res['dzdx'][i + 1])
        
        elif l['type'] == 'log':
            res['dzdx'][i] = vl_mylog(res['x'][i], res['dzdx'][i + 1])
        
        elif l['type'] == 'softmaxloss':
            res['dzdx'][i] = vl_mysoftmaxloss(res['x'][i], l['class'], res['dzdx'][i + 1])
        
        elif l['type'] == 'custom':
            res['dzdx'][i] = l.backward(l, res)

        else:
            raise Exception("Unknown layer type")
    
    return res
