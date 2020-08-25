import nnabla as nn
import numpy as np
import nnabla.functions as F
import nnabla.parametric_functions as PF
from nnabla.utils.data_iterator import data_iterator
from nnabla.utils.data_source import DataSource
import nnabla.solvers as S
from collections import OrderedDict

import utils_functions as UF
from data import *
from loss import *
import cv2
from tqdm import tqdm


from nnabla.models.imagenet import ResNet18

def get_output(f):
    if f.name=='BatchNormalization':
        stat = {}
        outs.append(f.inputs[0])
        stat['running_mean'] = f.inputs[3]
        print('neg', np.sum(f.inputs[4].d < 0))
        stat['running_std'] = (f.inputs[4] + 1e-6)**0.5
        batch_stats.append(stat)


def data_distill(uniform_data_iterator, num_iter):
    generated_img = []
    solver = S.Adam(alpha=0.5)
    for _ in range(uniform_data_iterator.size):
        dst_img = nn.Variable((bsize, 3, 224, 224), need_grad=True)
        img, _ = uniform_data_iterator.next()
        dst_img.d = img
        img_params = OrderedDict()
        img_params['img'] = dst_img
        solver.set_parameters(img_params)

        dummy_solver = S.Sgd(lr=1e-3)
        dummy_solver.set_parameters(nn.get_parameters())

        for it in tqdm(range(num_iter)):
            print(it)
            global outs
            outs = []
            global batch_stats
            batch_stats = []

            y = model(dst_img, force_global_pooling=True, training=False)
            y.forward(function_post_hook=get_output)
            assert len(outs) == len(batch_stats)
            loss = zeroq_loss(batch_stats, outs, dst_img)
            loss.forward()
            solver.zero_grad()
            dummy_solver.zero_grad()
            loss.backward()
            solver.weight_decay(1e-6)
            solver.update()
        generated_img.append(dst_img.d)

    return generated_img

def save_generated_img(generated_img, save_path):
    for index, batch_img in enumerate(generated_img):
        bsize = batch_img.shape[0]
        for i in range(bsize):
            img = batch_img[i].transpose((1,2,0))
            cv2.imwrite(f'{save_path}/{index*bsize+i}.png', img)



if __name__ == '__main__':
    model = ResNet18()
    data_length = 2
    uniform_data_source = UniformData(length=data_length, train=True, shuffle=True, rng=None)
    bsize = 2
    uniform_data_iterator = data_iterator(uniform_data_source, 
                                        batch_size=bsize, 
                                        rng=None, 
                                        with_memory_cache=False,
                                        with_file_cache=False)

    generated_img = data_distill(uniform_data_iterator, 2)
    save_generated_img(generated_img, 'generated')

    