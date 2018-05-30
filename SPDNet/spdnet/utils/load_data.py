import os
from scipy import io
import numpy as np

def load_data(data_aim = 'train'):
    """
        Load data from matlab
    """
    global_path = os.getcwd() + '/data/afew/spdface_400_inter_histeq/' + data_aim
    
    num_classes = 7
    
    data = []
    labels = []
    
    for num in range(1, num_classes + 1):
        
        path = global_path + '/' + str(num)
        
        names = sorted(os.listdir(path))
        
        for name in names:
            
            cur_mat = io.loadmat(path + '/' + name)
            
            data.append(cur_mat['Y1'])
            labels.append(num)


    data = np.array(data)
    labels = np.array(labels)

    return(data, labels)
