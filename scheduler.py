import warnings
import nnabla.utils.learning_rate_scheduler as lr_scheduler

class ReduceLROnPlateauScheduler(lr_scheduler.BaseLearningRateScheduler):
    '''
    nnabla implementation of https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#ReduceLROnPlateau
    usage:
        ```
        scheduler = ReduceLROnPlateauScheduler(init_lr=0.1)
        solver = S.Sgd(lr=0.1)
        for epoch in range(epochs):
            lr = scheduler.get_learning_rate()
            solver.set_learning_rate(lr)
            train(...)
            val_loss = valid(...)
            scheduler.update_lr(val_loss.d)
        ```
    '''

    def __init__(self, init_lr, mode='min', factor=0.1, patience=10, 
                 verbose=False, threshold=1e-4, threshold_mode='rel',
                 cooldown=0, min_lr=0, eps=1e-8):
        self.lr = init_lr
        self.min_lr = min_lr
        if factor >= 1.0:
            raise ValueError('Factor should be < 1.0.')
        self.factor = factor
        
        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0
        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.best = None
        self.num_bad_epochs = None
        self.mode_worse = None  # the worse value for the chosen mode
        self.eps = eps
        self.last_epoch = 0
        self._init_is_better(mode=mode, threshold=threshold,
                             threshold_mode=threshold_mode)
        self._reset()
        
        self.is_lr_updated = True # initialize with True
        
    def _reset(self):
        """Resets num_bad_epochs counter and cooldown counter."""
        self.best = self.mode_worse
        self.cooldown_counter = 0
        self.num_bad_epochs = 0
        
    def update_lr(self, metrics, epoch=None):
        # convert `metrics` to float, in case it's a zero-dim Variable
        current = float(metrics)
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0  # ignore any bad epochs in cooldown

        if self.num_bad_epochs > self.patience:
            self._reduce_lr(epoch)
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0

        self.is_lr_updated = True

    def _reduce_lr(self, epoch):
        old_lr = self.lr
        new_lr = new_lr = max(old_lr * self.factor, self.min_lr)
        if old_lr - new_lr > self.eps:
            self.lr = new_lr
            if self.verbose:
                print('Epoch {:5d}: reducing learning rate'
                      'to {:.4e}.'.format(epoch, new_lr))
                
    def get_learning_rate(self):
        if not self.is_lr_updated:
            warnings.warn('Please update learning rate before get learning rate', RuntimeWarning)
        self.is_lr_updated = False 
        return self.lr
        
    @property
    def in_cooldown(self):
        return self.cooldown_counter > 0

    def is_better(self, a, best):
        if self.mode == 'min' and self.threshold_mode == 'rel':
            rel_epsilon = 1. - self.threshold
            return a < best * rel_epsilon

        elif self.mode == 'min' and self.threshold_mode == 'abs':
            return a < best - self.threshold

        elif self.mode == 'max' and self.threshold_mode == 'rel':
            rel_epsilon = self.threshold + 1.
            return a > best * rel_epsilon

        else:  # mode == 'max' and epsilon_mode == 'abs':
            return a > best + self.threshold
        
    def _init_is_better(self, mode, threshold, threshold_mode):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if threshold_mode not in {'rel', 'abs'}:
            raise ValueError('threshold mode ' + threshold_mode + ' is unknown!')

        if mode == 'min':
            self.mode_worse = float('inf') 
        else:  # mode == 'max':
            self.mode_worse = -float('inf') 

        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode