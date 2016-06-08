import numpy as np


def std_normalize(data, mean=0, std=1, inplace=False):
    xs_mean = np.mean(data.xs, axis=0)
    xs_std = np.std(data.xs, axis=0)
    data.config = {'normalize_method': 'std', 'xs_mean': xs_mean, 'xs_std': xs_std, 'mean': mean, 'std': std}
    xs = (data.xs - xs_mean)*(std / xs_std) + mean
    xs[np.isnan(xs)] = 0
    if inplace:
        data.xs = xs
        # data.ys = ys
    else:
        norm_data = data.copy()
        norm_data.xs = xs
        # norm_data.ys = ys
        return norm_data


def minmax_normalize(data, lower_bound=-1, upper_bound=1, inplace=False):
    xs_max = np.max(data.xs, axis=0)
    xs_min = np.min(data.xs, axis=0)
    data.config = {'normalize_method': 'minmax', 'xs_max': xs_max, 'xs_min': xs_min,
                   "lower_bound": lower_bound, 'upper_bound': upper_bound}
    # ys_max = np.max(data.ys, axis=1)[:, np.newaxis]
    # ys_min = np.min(data.ys, axis=1)[:, np.newaxis]
    xs = (upper_bound - lower_bound) * (data.xs - xs_min) / (xs_max - xs_min) + lower_bound
    xs[np.isnan(xs)] = 0
    # ys = (data.ys - ys_min) / (ys_max - ys_min)
    if inplace:
        data.xs = xs
        # data.ys = ys
    else:
        norm_data = data.copy()
        norm_data.xs = xs
        # norm_data.ys = ys
        return norm_data


def mean_normalize(data, inplace=False):
    xs_mean = np.mean(data.xs, axis=0)
    # ys_mean = np.mean(data.ys, axis=1)[:, np.newaxis]
    xs_max = np.max(data.xs, axis=0)
    xs_min = np.min(data.xs, axis=0)
    data.config = {'normalize_method': 'mean', 'xs_max': xs_max, 'xs_min': xs_min, 'mean': xs_mean}
    # ys_max = np.max(data.ys, axis=1)[:, np.newaxis]
    # ys_min = np.min(data.ys, axis=1)[:, np.newaxis]
    xs = (data.xs - xs_mean) / (xs_max - xs_min)
    xs[np.isnan(xs)] = 0
    # ys = (data.ys - ys_mean)/(ys_max - ys_min)
    # ys = data.ys
    if inplace:
        data.xs = xs
        # data.ys = ys
    else:
        norm_data = data.copy()
        norm_data.xs = xs
        # norm_data.ys = ys
        return norm_data