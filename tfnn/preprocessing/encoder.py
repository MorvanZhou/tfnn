import numpy as np
import pandas as pd


class BinaryEncoder(object):

    def encode_target(self, data, columns=None, inplace=False):
        """
        1-of-C dummy-coding the categorical target data.
        :param data:
        :param columns: columns to be converted.
        :param inplace:
        :return:
        """
        result = self.numpy_1_of_k(data.ys, )
        if inplace:
            data.ys = result
            return None
        else:
            return result

    @staticmethod
    def encode_data(data, columns=None, inplace=False):
        """
        1-of-(C-1) effects-coding the categorical feature data.
        :param data:
        :param columns: columns to be converted.
        :param inplace:
        :return:
        """
        result = pd.get_dummies(data.xs, columns=columns, drop_first=True)  # drop_first exist in pandas >= 0.18.1 only
        # to be done: convert C-1 all 0 data to all -1
        if inplace:
            data.xs = result
            return None
        else:
            return result

    @staticmethod
    def numpy_1_of_k(seq):
        unique_num, indices, n_classes = np.unique(seq, return_inverse=True, return_counts=True)
        n_samples = len(seq)
        one_hot = np.zeros((n_samples, n_classes))
        one_hot[np.arange(n_samples), indices] = 1
        return one_hot


if __name__ == '__main__':
    def numpy_1_of_k(seq):
        unique_num, indices= np.unique(seq, return_inverse=True)
        n_samples = len(seq)
        n_classes = len(unique_num)
        one_hot = np.zeros((n_samples, n_classes))
        one_hot[np.arange(n_samples), indices] = 1
        return one_hot

    d = np.array([[1,1],[2,2]])
    print(numpy_1_of_k(d))

