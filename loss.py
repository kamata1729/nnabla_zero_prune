
import math
import nnabla as nn
import numpy as np
import nnabla.functions as F
import utils_functions as UF


def own_loss(A, B):
    """
    L-2 loss between A and B normalized by length.
    Shape of A should be (1, features_num), shape of B should be (batch_size, features_num)
    """
    return F.sum((A-B)**2) / B.shape[0]


def zeroq_loss(batch_stats, outs, random_input):
    # batch stats loss
    batch_mean_loss = 0 
    batch_std_loss = 0
    for cnt, (bn_stat, out) in enumerate(zip(batch_stats, outs)):
        bn_mean = bn_stat['running_mean'][:, :, 0, 0] # (1, channels)
        bn_std = bn_stat['running_std'][:, :, 0, 0]# (1, channels)
        out_mean = F.mean(out, axis=(2,3))# (batch, channels)
        out_std = UF.std(out, axis=(2,3))# (batch, channels)
        """
        tmp = own_loss(bn_mean, out_mean)
        tmp.forward()
        print('mean', tmp.d)
        
        tmp =own_loss(bn_std, out_std)
        tmp.forward()
        print('std', tmp.d)
        """
        
        batch_mean_loss += own_loss(bn_mean, out_mean)
        batch_std_loss += own_loss(bn_std, out_std)
        
    # input loss
    
    input_mean = nn.Variable.from_numpy_array(np.zeros((1,3)))
    input_std = nn.Variable.from_numpy_array(np.ones((1,3)))
    random_input_mean = F.mean(random_input, axis=(2,3)) # (batch, channels)
    random_input_std = UF.std(random_input, axis=(2,3)) # (batch, channels)

    
    input_mean_loss = own_loss(input_mean, random_input_mean)
    input_std_loss = own_loss(input_std, random_input_std)
    
    batch_mean_loss.forward()
    input_mean_loss.forward()
    print('batch_mean_loss', batch_mean_loss.d, 'input_mean_loss', input_mean_loss.d)

    total_loss = batch_mean_loss + batch_std_loss + input_mean_loss + input_std_loss
    return total_loss