import numpy as np
import pandas as pd
import tfnn
from tfnn.datasets.shuffle import shuffle


def train_test_split(data, train_rate=0.7, randomly=True):
    _n_train_samples = int(data.n_samples * train_rate)
    if randomly:
        xs_ys = pd.concat([data.xs, data.ys], axis=1, join='outer')
        df = xs_ys.reindex(np.random.permutation(xs_ys.index))
        shuffled_xs = df.iloc[:, :data.xs.shape[1]]
        shuffled_ys = df.iloc[:, data.xs.shape[1]:]
        t_xs = shuffled_xs.iloc[:_n_train_samples, :]
        t_ys = shuffled_ys.iloc[:_n_train_samples, :]
        v_xs = shuffled_xs.iloc[_n_train_samples:, :]
        v_ys = shuffled_ys.iloc[_n_train_samples:, :]
    else:
        t_xs = data.xs.iloc[_n_train_samples:, :]
        t_ys = data.ys.iloc[_n_train_samples:, :]
        v_xs = data.xs.iloc[_n_train_samples:, :]
        v_ys = data.ys.iloc[_n_train_samples:, :]
    t_data = tfnn.Data(t_xs, t_ys, name='train')
    v_data = tfnn.Data(v_xs, v_ys, name='validate')
    return [t_data, v_data]