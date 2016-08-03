import pandas as pd


def sampled_batch(data, batch_size, replace=False, random_state=None):
    """

    :param data:
    :param batch_size:
    :param replace: Allow replacements in sampled data
    :param random_state:
    :return:
    """
    xs_ys = pd.concat([data.xs, data.ys], axis=1)
    batch = xs_ys.sample(n=batch_size, axis=0, replace=replace, random_state=random_state)
    batch_xs = batch.iloc[:, :data.xs.shape[1]]
    batch_ys = batch.iloc[:, data.xs.shape[1]:]
    return [batch_xs, batch_ys]
