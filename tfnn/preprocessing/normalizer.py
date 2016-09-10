import numpy as np


class Normalizer(object):
    def __init__(self):
        self.config_exist = False
        self.config = None

    def set_config(self, data_config):
        self.config = data_config
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
        self.config_exist = True

    def fit_transform(self, xs):
        """
        When having instant test, use fit_transform to normalize one sample xs data
        :param xs:
        :return: normalized xs
        """
        if self.config_exist:
            if self.method == 'minmax':
                n_xs = (self.upper_bound - self.lower_bound) * (xs - self.xs_min) \
                     / (self.xs_max - self.xs_min) + self.lower_bound
                n_xs[np.isnan(n_xs)] = 0
            elif self.method == 'mean':
                n_xs = (xs - self.xs_mean) / (self.xs_max - self.xs_min)
                n_xs[np.isnan(n_xs)] = 0
            elif self.method == 'std':
                n_xs = (xs - self.xs_mean) * (self.std / self.xs_std) + self.mean
                n_xs[np.isnan(n_xs)] = 0
            return n_xs
        else:
            raise AttributeError('Have not set normalizer config')

    def minmax(self, data, lower_bound=-1, upper_bound=1, inplace=False):
        xs_max = np.max(data.xs, axis=0)
        xs_min = np.min(data.xs, axis=0)
        config = {'normalize_method': 'minmax', 'xs_max': xs_max, 'xs_min': xs_min,
                  "lower_bound": lower_bound, 'upper_bound': upper_bound}
        self.set_config(config)
        xs = (self.upper_bound - self.lower_bound) * (data.xs - self.xs_min) \
             / (self.xs_max - self.xs_min) + self.lower_bound
        xs[np.isnan(xs)] = 0
        return self._check_inplace(data, xs, inplace)

    def std(self, data, mean=0, std=1, inplace=False):
        xs_mean = np.mean(data.xs, axis=0)
        xs_std = np.std(data.xs, axis=0)
        config = {'normalize_method': 'std', 'xs_mean': xs_mean, 'xs_std': xs_std, 'mean': mean, 'std': std}
        self.set_config(config)
        xs = (data.xs - self.xs_mean) * (self.std / self.xs_std) + self.mean
        return self._check_inplace(data, xs, inplace)

    def mean(self, data, inplace=False):
        xs_mean = np.mean(data.xs, axis=0)
        # ys_mean = np.mean(data.ys, axis=1)[:, np.newaxis]
        xs_max = np.max(data.xs, axis=0)
        xs_min = np.min(data.xs, axis=0)
        config = {'normalize_method': 'mean', 'xs_max': xs_max, 'xs_min': xs_min, 'mean': xs_mean}
        self.set_config(config)
        xs = (data.xs - self.xs_mean) / (self.xs_max - self.xs_min)
        xs[np.isnan(xs)] = 0
        return self._check_inplace(data, xs, inplace)

    def _check_inplace(self, data, xs, inplace):
        if inplace:
            data.data[:, :data.n_xfeatures] = xs
        else:
            norm_data = data.copy()
            norm_data.data[:, :norm_data.n_xfeatures] = xs
            return norm_data