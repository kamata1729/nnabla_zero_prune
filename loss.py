
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



def input_loss_imagenet(input_data):

    imagenet_mean=nn.Variable.from_numpy_array(np.array([[0.485, 0.456, 0.406]]))
    imagenet_std=nn.Variable.from_numpy_array(np.array([[0.229, 0.224, 0.225]]))
    
    input_data_mean = F.mean(input_data, axis=(2,3)) # (batch, channels)
    input_data_std = UF.std(input_data, axis=(2,3)) # (batch, channels)

    input_mean_loss = own_loss(imagenet_mean, input_data_mean)
    input_std_loss = own_loss(imagenet_std, input_data_std)
    """
    input_data = (input_data * 0.01735) - 1.99 # normalize method of imagenet pretrained model
    input_data_mean = F.mean(input_data, axis=(2,3)) # (batch, channels)
    input_data_std = UF.std(input_data, axis=(2,3)) # (batch, channels)

    uniform_mean = (0 + 255) / 2
    normalized_mean = (uniform_mean * 0.01735) - 1.99
    pretrained_mean = nn.Variable.from_numpy_array(np.ones((1,3)) * normalized_mean)
    
    uniform_std = math.sqrt((255 - 0)**2 / 12)
    normalized_std = uniform_std * 0.01735
    pretrained_std = nn.Variable.from_numpy_array(np.ones((1,3)) * normalized_std)
    
    input_mean_loss = own_loss(pretrained_mean, input_data_mean)
    input_std_loss = own_loss(pretrained_std, input_data_std)
    """
    return input_mean_loss, input_std_loss

def zeroq_loss(batch_stats, outs, random_input):
    # batch stats loss
    mean_loss = 0 
    std_loss = 0
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
        
        mean_loss += own_loss(bn_mean, out_mean)
        std_loss += own_loss(bn_std, out_std)
        
    # input loss
    
    input_mean = nn.Variable.from_numpy_array(np.ones((1,3)))
    input_std = nn.Variable.from_numpy_array(np.ones((1,3)))
    random_input_mean = F.mean(random_input, axis=(2,3)) # (batch, channels)
    random_input_std = UF.std(random_input, axis=(2,3)) # (batch, channels)

    
    mean_loss += own_loss(input_mean, random_input_mean)
    std_loss += own_loss(input_std, random_input_std)
    """
    input_mean_loss, input_std_loss = input_loss_imagenet(random_input)
    mean_loss += input_mean_loss
    std_loss += input_std_loss
    """
    total_loss = mean_loss + std_loss
    return total_loss