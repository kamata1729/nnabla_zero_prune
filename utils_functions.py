
import nnabla as nn
import numpy as np
import nnabla.functions as F
import nnabla.parametric_functions as PF

def var(inp, axis=None):
    mean = F.mean(inp, axis=axis, keepdims=True)
    out = F.mean((inp-mean)**2, axis=axis)
    return out

def std(inp, axis=None, eps=1e-6):
    out = (var(inp, axis)+eps) ** 0.5
    return out

def sqrt(inp):
    return inp ** 0.5