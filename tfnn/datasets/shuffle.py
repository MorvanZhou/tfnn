import numpy as np
import pandas as pd


def shuffle(data, inplace=False):
    xs_ys = pd.concat([data.xs, data.ys], axis=1, join='outer')
    df = xs_ys.reindex(np.random.permutation(xs_ys.index))
    xs = df.iloc[:, :data.xs.shape[1]]
    ys = df.iloc[:, data.xs.shape[1]:]
    if inplace:
        data.xs, data.ys = xs, ys
        return None
    else:
        return [xs, ys]