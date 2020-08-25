
import nnabla as nn
import numpy as np
import nnabla.functions as F
import utils_functions as UF


def own_loss(A, B):
    """
    L-2 loss between A and B normalized by length.
    Shape of A should be (features_num, ), shape of B should be (batch_size, features_num)
    """
    return F.mean((A-B)**2, axis=1)

def zeroq_loss(batch_stats, outs, random_input):
    # batch stats loss
    mean_loss = 0 
    std_loss = 0
    for cnt, (bn_stat, out) in enumerate(zip(batch_stats, outs)):
        bn_mean = bn_stat['running_mean'][:, :, 0, 0] # (channels, )
        bn_std = bn_stat['running_std'][:, :, 0, 0]# (channels, )
        out_mean = F.mean(out, axis=(2,3))# (batch, channels)
        out_std = std(out, axis=(2,3))# (batch, channels)
        mean_loss += own_loss(bn_mean, out_mean)
        std_loss += own_loss(bn_std, out_std)
        
    # input loss
    input_mean = nn.Variable.from_numpy_array(np.ones((1,3)))
    input_std = nn.Variable.from_numpy_array(np.ones((1,3)))
    random_input_mean = F.mean(random_input, axis=(2,3)) # (batch, channels)
    random_input_std = std(random_input, axis=(2,3))
    
    mean_loss += own_loss(input_mean, random_input_mean)
    std_loss += own_loss(input_std, random_input_std)
    total_loss = mean_loss + std_loss
    return total_loss