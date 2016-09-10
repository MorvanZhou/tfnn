import numpy as np


def sampled_batch(data, batch_size, replace=False, p=None):
    """

    :param data:
    :param batch_size:
    :param replace: Allow replacements in sampled data
    :param p : 1-D array-like, optional
                The probabilities associated with each entry in a.
                If not given the sample assumes a uniform distribution over all entries in a.
    :return:
    """
    _sampled_batch = data.data[np.random.choice(data.data.shape[0], batch_size, replace=replace, p=p), :]

    return [_sampled_batch[:, :data.n_xfeatures], _sampled_batch[:, data.n_xfeatures:]]
