import numpy as np
import tfnn


def train_test_split(data, train_rate=0.7, randomly=True):
    _n_train_samples = int(data.n_samples * train_rate)
    if randomly:
        n_x = data.xs.shape[1]
        xs_ys = np.hstack((data.xs, data.ys))
        np.random.shuffle(xs_ys)
        shuffled_xs = xs_ys[:, :n_x]
        shuffled_ys = xs_ys[:, n_x:]
        t_xs = shuffled_xs[:_n_train_samples, :]
        t_ys = shuffled_ys[:_n_train_samples, :]
        v_xs = shuffled_xs[_n_train_samples:, :]
        v_ys = shuffled_ys[_n_train_samples:, :]
    else:
        t_xs = data.xs[_n_train_samples:, :]
        t_ys = data.ys[_n_train_samples:, :]
        v_xs = data.xs[_n_train_samples:, :]
        v_ys = data.ys[_n_train_samples:, :]
    t_data = tfnn.Data(t_xs, t_ys, name='train')
    v_data = tfnn.Data(v_xs, v_ys, name='validate')
    return [t_data, v_data]