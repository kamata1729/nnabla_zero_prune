import nnabla as nn
import numpy as np
from nnabla.utils.data_source import DataSource

class UniformData(DataSource):
    
    def _get_data(self, position):
        image = self._images[self._indexes[position]]
        label = self._labels[self._indexes[position]]
        return (image, label)
    
    def __init__(self, length, train=True, shuffle=False, rng=None):
        super().__init__(shuffle=shuffle)
        self._train = train
        self._images = np.random.randint(0,255,(length, 3, 224, 224))
        # var[U(-128, 127)] = (127 - (-128))**2 / 12 = 5418.75
        self._images = (self._images.astype(np.float) - 127.5) / np.sqrt(5418.75) # この辺も怪しい
        self._labels = np.random.randint(0, 1000, length).reshape(-1, 1)
        self._size = self._labels.size
        self._variables = ('x', 'y')
        if rng is None:
            rng = np.random.RandomState(313)
        self.rng = rng
        self.reset()
        
    def reset(self):
        if self._shuffle:
            self._indexes = self.rng.permutation(self._size)
        else:
            self._indexes = np.arange(self._size)
        super().reset()
    
    @property
    def images(self):
        """Get copy of whole data with a shape of (N, 1, H, W)."""
        return self._images.copy()

    @property
    def labels(self):
        """Get copy of whole label with a shape of (N, 1)."""
        return self._labels.copy()

