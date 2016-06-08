import numpy as np


class NormalizeFilter(object):
    def __init__(self, data_config):
        self.method = data_config['normalize_method']
        if self.method == 'minmax':
            self.xs_max = data_config['xs_max']
            self.xs_min = data_config['xs_min']
            self.lower_bound = data_config["lower_bound"]
            self.upper_bound = data_config['upper_bound']
        elif self.method == 'mean':
            self.xs_max = data_config['xs_max']
            self.xs_min = data_config['xs_min']
            self.xs_mean = data_config['mean']
        elif self.method == 'std':
            self.xs_mean = data_config['xs_mean']
            self.xs_std = data_config['xs_std']
            self.mean = data_config['mean']
            self.std = data_config['std']

    def filter(self, xs):
        if self.method == 'minmax':
            xs = (self.upper_bound - self.lower_bound) * (xs - self.xs_min) \
                 / (self.xs_max - self.xs_min) + self.lower_bound
            xs[np.isnan(xs)] = 0
        elif self.method == 'mean':
            xs = (xs - self.xs_mean) / (self.xs_max - self.xs_min)
            xs[np.isnan(xs)] = 0
        elif self.method == 'std':
            xs = (xs - self.xs_mean) * (self.std / self.xs_std) + self.mean
            xs[np.isnan(xs)] = 0
        return xs
