import numpy as np
import pandas as pd


class BinaryEncoder(object):
    @staticmethod
    def encode_target(data, columns=None, inplace=False):
        """
        1-of-C dummy-coding the categorical target data.
        :param data:
        :param columns: columns to be converted.
        :param inplace:
        :return:
        """
        result = pd.get_dummies(data.ys, columns=columns)

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
        result = pd.get_dummies(data.ys, columns=columns, drop_first=True)  # drop_first exist in pandas >= 0.18.1 only
        # to be done: convert C-1 all 0 data to all -1
        if inplace:
            data.ys = result
            return None
        else:
            return result


if __name__ == '__main__':
    class Data:
        xs = pd.DataFrame({'a': ['d', 'f', 'm', 'm'], 'b': [1.2, 2, 3, 4], 'c': ['d','a','a','d']})
        ys = pd.DataFrame({'a': ['m', 'f', 'f', 's']})
        n_samples = 4

    data = Data()
    print(data.xs.dtypes)
    print(pd.get_dummies(data.xs, columns=['a'], drop_first=True))
