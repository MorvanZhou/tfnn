import numpy as np
import copy

from tfnn.preprocessing.shuffle import shuffle as datasets_shuffle
from tfnn.preprocessing.train_test_split import train_test_split as datasets_train_test_split
from tfnn.preprocessing.onehot_encode import onehot_encode as datasets_onehot_encode
from tfnn.preprocessing.encoder import BinaryEncoder
from tfnn.preprocessing.sampled_batch import sampled_batch as datasets_sampled_batch
from tfnn.preprocessing.plot_feature_utility import plot_feature_utility as datasets_plot_feature_utility
from tfnn.preprocessing.next_batch import next_batch as datasets_next_batch


class Data:
    def __init__(self, xs, ys, name=None):
        """
        Input data sets.
        :param xs: data, shape(n_samples, n_xs), accept numpy, pandas, list
        :param ys: labels, shape(n_samples, n_ys), accept numpy, pandas, list
        """
        if (type(xs).__module__ == np.__name__) & (type(ys).__module__ == np.__name__):
            self.module = 'numpy_data'
        elif ('pandas' in type(xs).__module__) & ('pandas' in type(ys).__module__):
            xs, ys = np.asarray(xs), np.asarray(ys)
        elif (type(xs) == list) & (type(ys) == list):
            xs, ys = np.asarray(xs), np.asarray(ys)
        else:
            raise TypeError('all data type must be numpy or pandas')
        if ys.ndim < 2:
            ys = ys[:, np.newaxis]
        if xs.ndim < 2:
            xs = xs[:, np.newaxis]

        self.n_xfeatures = xs.shape[-1]     # col for 2 dims, channel for 3 dims
        self.n_yfeatures = ys.shape[-1]     # col for 2 dims,
        self.data = np.hstack((xs, ys))
        self.n_samples = ys.shape[0]
        self.name = name

    @property
    def xs(self):
        return self.data[:, :self.n_xfeatures]

    @property
    def ys(self):
        return self.data[:, self.n_xfeatures:]

    def shuffle(self, inplace=False):
        _shuffled_data = datasets_shuffle(self)
        if inplace:
            self.data = _shuffled_data.data
        else:
            return _shuffled_data

    def onehot_encode_y(self, inplace=False):
        """
        1-of-C dummy-coding the categorical target data.
        :param inplace: True of False
        :return:
        """
        _ys = datasets_onehot_encode(self.ys, inplace)
        data_copy = self.data.copy()
        data_copy = np.delete(data_copy, self.n_xfeatures, axis=1)
        data_copy = np.insert(data_copy, [self.n_xfeatures], _ys, axis=1)
        if inplace:
            self.data = data_copy
        else:
            _xs = data_copy[:, :self.n_xfeatures]
            _ys = data_copy[:, self.n_xfeatures:]
            return Data(_xs, _ys)

    def sampled_batch(self, batch_size, replace=False, p=None):
        """

        :param batch_size:
        :param replace: Allow replacements in sampled data
        :param p : 1-D array-like, optional
                The probabilities associated with each entry in a.
                If not given the sample assumes a uniform distribution over all entries in a.
        :return:
        """
        return datasets_sampled_batch(self, batch_size, replace, p)

    def next_batch(self, batch_size):
        return datasets_next_batch(self, batch_size)

    def plot_feature_utility(self, n_feature):
        """
        This function is to check the categorical feature utility for machine learning BEFORE BINARIZE.
        :param selected_feature_name:
        :param target_name:
        :return:
        """
        datasets_plot_feature_utility(self, n_feature)

    def train_test_split(self, train_rate=0.7, randomly=True):
        t_data, v_data = datasets_train_test_split(self, train_rate, randomly)
        return [t_data, v_data]

    def copy(self):
        deep_copy = copy.deepcopy([self.xs, self.ys, self.name])
        return Data(*deep_copy)

if __name__ == "__main__":
    import pandas as pd
    xs = pd.DataFrame({'a': ['a','d','f','f','a'],
                       'b': [1,2,3,4,5],
                       'c': ['f','m','m','f','f']})
    ys = pd.Series(['y','n','y','y','n'])
    xs = np.arange(12).reshape((4,3))
    ys = np.array(['y','n','y', 'y'])
    data = Data(xs, ys)
    print(data.xs, '\n', data.ys)
    t_data  = data.shuffle(inplace=False)
    print(t_data.xs,'\n', t_data.ys)