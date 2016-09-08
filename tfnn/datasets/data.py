import numpy as np
import pandas as pd
import copy

from tfnn.datasets.shuffle import shuffle as data_sets_shuffle
from tfnn.datasets.train_test_split import train_test_split as data_sets_train_test_split
from tfnn.datasets.to_binary import BinaryEncoder
from tfnn.datasets.sampled_batch import sampled_batch as data_sets_sampled_batch
from tfnn.datasets.plot_feature_utility import plot_feature_utility as data_sets_plot_feature_utility
from tfnn.datasets.next_batch import next_batch as data_sets_next_batch


class Data:
    def __init__(self, xs, ys, name=None):
        """
        Input data sets.
        :param xs: data, shape(n_xs, n_samples), (pd.DataFrame)
        :param ys: labels, shape(n_ys, n_samples), (pd.DataFrame)
        """
        # TODO: try only use numpy and tfnn tensor
        if ('numpy' in type(xs).__module__) & ('numpy' in type(ys).__module__):
            xs = pd.DataFrame(xs)
            ys = pd.DataFrame(ys)
        elif ('pandas' in type(xs).__module__) & ('pandas' in type(ys).__module__):
            self.module = 'pandas'
        else:
            raise TypeError('data type must be numpy or pandas')
        if type(xs) is pd.core.series.Series:
            xs = xs.to_frame()
        if type(ys) is pd.core.series.Series:
            ys = ys.to_frame()
        self.xs = xs.copy()  # shape (n_xs, n_samples)
        self.ys = ys.copy()  # shape (n_ys, n_samples)
        self.n_samples = ys.shape[0]
        self.name = name

    def shuffle(self, inplace=False):
        result = data_sets_shuffle(self, inplace)
        if result is not None:
            return result

    def encode_cat_y(self, columns=None, inplace=False):
        """
        1-of-C dummy-coding the categorical target data.
        :param inplace: True of False
        :return:
        """
        encoder = BinaryEncoder()
        result = encoder.encode_target(self, columns, inplace)
        if result is not None:
            return result

    def encode_cat_x(self, columns=None, inplace=False):
        """
        1-of-(C-1) effects-coding the categorical feature data.
        :features_name: If None, encode all features. Otherwise features_name should be given as an list,
        eg. ['featrue1','feature2'].
        :param inplace: True or False
        :return:
        """
        encoder = BinaryEncoder()
        result = encoder.encode_data(self, columns, inplace)
        if result is not None:
            return result

    def sampled_batch(self, batch_size, replace=False, random_state=None):
        """

        :param batch_size:
        :param replace: Allow replacements in sampled data
        :param random_state:
        :return:
        """
        return data_sets_sampled_batch(self, batch_size, replace, random_state)

    def next_batch(self, batch_size, loop=False):
        return data_sets_next_batch(self, batch_size, loop)

    def plot_feature_utility(self, selected_feature_name, target_name=None):
        """
        This function is to check the categorical feature utility for machine learning BEFORE BINARIZE.
        :param selected_feature_name:
        :param target_name:
        :return:
        """
        data_sets_plot_feature_utility(self, selected_feature_name, target_name)

    def train_test_split(self, train_rate=0.7, randomly=True):
        t_data, v_data = data_sets_train_test_split(self, train_rate, randomly)
        return [t_data, v_data]

    def copy(self):
        return copy.deepcopy(self)


if __name__ == "__main__":
    xs = pd.DataFrame({'a': ['a','d','f','f','a'],
                       'b': [1,2,3,4,5],
                       'c': ['f','m','m','f','f']})
    ys = pd.DataFrame({'answer': ['y','n','y','y','n']})
    # xs = np.arange(12).reshape((4,3))
    # ys = np.array(['y','n','y', 'y'])
    data = Data(xs, ys)
    train, test = data.train_test_split()
    print(test.xs)
    print(test.ys)