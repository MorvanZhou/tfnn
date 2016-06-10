import numpy as np
import copy

from tfnn.datasets.normalize import std_normalize as data_sets_std_normalize
from tfnn.datasets.normalize import minmax_normalize as data_sets_minmax_normalize
from tfnn.datasets.shuffle import shuffle as data_sets_shuffle
from tfnn.datasets.train_test_split import train_test_split as data_sets_train_test_split
from tfnn.datasets.to_binary import to_binary as data_sets_to_binary
from tfnn.datasets.sampled_batch import sampled_batch as data_sets_sampled_batch


class Data:
    def __init__(self, xs, ys, name=None):
        """
        Input data sets.
        :param xs: data, shape(n_xs, n_samples), (pd.DataFrame)
        :param ys: labels, shape(n_ys, n_samples), (pd.DataFrame)
        """

        if 'pandas' in type(xs).__module__:
            xs = xs.as_matrix()
        if 'pandas' in type(ys).__module__:
            ys = ys.as_matrix()

        if xs.ndim == 1:
            xs = xs[:, np.newaxis]
        if ys.ndim == 1:
            ys = ys[:, np.newaxis]
        xs_type, ys_type = type(xs).__module__, type(ys).__module__
        if xs_type == np.__name__:
            if ys_type == np.__name__:
                self.xs = xs.copy()  # shape (n_xs, n_samples)
                self.ys = ys.copy()  # shape (n_ys, n_samples)
            else:
                raise ValueError('Data have to be numpy or pandas.core.frame')
        else:
            raise ValueError('Data have to be numpy or pandas.core.frame')
        self.n_samples = ys.shape[0]
        self.name = name

    def std_normalize(self, mean=0, std=1, inplace=False):
        if inplace:
            data_sets_std_normalize(self, mean, std, inplace)
        else:
            return data_sets_std_normalize(self, mean, std, inplace)

    def minmax_normalize(self, lower_bound=-1, upper_bound=1, inplace=False):
        if inplace:
            data_sets_minmax_normalize(self, lower_bound, upper_bound, inplace)
        else:
            return data_sets_minmax_normalize(self, lower_bound, upper_bound, inplace)

    def shuffle(self, inplace=False):
        if inplace:
            data_sets_shuffle(self, inplace)
        else:
            return data_sets_shuffle(self, inplace)

    def to_binary(self, inplace=False):
        if inplace:
            data_sets_to_binary(self, inplace)
        else:
            return data_sets_to_binary(self, inplace)

    def sampled_batch(self, batch_size):
        return data_sets_sampled_batch(self, batch_size)

    def next_batch(self, batch_size, loop=False):
        try:
            self._batch_index_segments
        except AttributeError:
            self._n_segments = self.n_samples//batch_size
            self._batch_index_segments = np.array_split(np.arange(self.n_samples), self._n_segments)
            self._batch_counter = 0
        if loop:
            self._batch_counter += 1
            _mini_batch_indexes = self._batch_index_segments[self._batch_counter - 1]
            batch_xs = self.xs[_mini_batch_indexes, :]
            batch_ys = self.ys[_mini_batch_indexes, :]
            if self._batch_counter == self._n_segments:
                self._batch_counter = 0
        else:
            self._batch_counter += 1
            _mini_batch_indexes = self._batch_index_segments[self._batch_counter - 1]
            batch_xs = self.xs[_mini_batch_indexes, :]
            batch_ys = self.ys[_mini_batch_indexes, :]
            if self._batch_counter == self._n_segments:
                raise IndexError('No more training data. Either set loop=True or reduce training steps.')
        return [batch_xs, batch_ys]

    def train_test_split(self, train_rate=0.7, randomly=True):
        t_data, v_data = data_sets_train_test_split(self, train_rate, randomly)
        return [t_data, v_data]

    def copy(self):
        return copy.deepcopy(self)