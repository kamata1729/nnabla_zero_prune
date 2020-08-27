import nnabla as nn
import numpy as np
import nnabla.functions as F
import nnabla.parametric_functions as PF
from nnabla.utils.data_iterator import data_iterator
from nnabla.utils.data_source import DataSource
import nnabla.utils.learning_rate_scheduler as lr_scheduler
from scheduler import *
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
        """
        f.inputs = [
            input feature,
            gamma of bn,
            beta of bn,
            running_mean,
            running_std
        ]
        """
        stat = {}
        outs.append(f.inputs[0])
        stat['running_mean'] = nn.Variable.from_numpy_array(f.inputs[3].d, need_grad=False)
        stat['running_std'] = nn.Variable.from_numpy_array((f.inputs[4].d + 1e-6)**0.5, need_grad=False) 
        batch_stats.append(stat)


def data_distill(model, uniform_data_iterator, num_iter):
    generated_img = []
    for _ in range(uniform_data_iterator.size // uniform_data_iterator.batch_size):
        img, _ = uniform_data_iterator.next()
        dst_img = nn.Variable(img.shape, need_grad=True)
        dst_img.d = img
        img_params = OrderedDict()
        img_params['img'] = dst_img

        init_lr = 2.5
        solver = S.Adam(alpha=init_lr)
        solver.set_parameters(img_params)
        #scheduler = lr_scheduler.CosineScheduler(init_lr=0.5, max_iter=num_iter)
        scheduler = ReduceLROnPlateauScheduler(init_lr=init_lr, min_lr=1e-4, verbose=False, patience=100)
        dummy_solver = S.Sgd(lr=0)
        dummy_solver.set_parameters(nn.get_parameters())

        for it in tqdm(range(num_iter)):
            lr = scheduler.get_learning_rate()
            solver.set_learning_rate(lr)

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

            scheduler.update_lr(loss.d)

        generated_img.append(dst_img.d)

    return generated_img

def save_generated_img(generated_img, save_path):
    for index, batch_img in enumerate(generated_img):
        bsize = batch_img.shape[0]
        for i in range(bsize):
            img = batch_img[i].transpose((1,2,0))
            img = np.clip(img * np.sqrt(5418.75) + 127.5, 0, 254).astype(np.uint8)
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

    generated_img = data_distill(model, uniform_data_iterator, 2)
    save_generated_img(generated_img, 'generated')


    