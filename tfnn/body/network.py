import numpy as np
import time
import tfnn
from tfnn.body.layer import Layer
from tfnn.preprocessing.normalizer import Normalizer


class Network(object):
    def __init__(self, input_size, output_size, do_dropout, do_l2, ntype):
        self.normalizer = Normalizer()
        self.input_size = input_size
        self.output_size = output_size
        self.global_step = tfnn.Variable(0, trainable=False)
        if do_dropout and do_l2:
            raise ValueError('Cannot do dropout and l2 at once. Choose only one of them.')
        if do_dropout:
            self.reg = 'dropout'
        if do_l2:
            self.reg = 'l2'
        if (do_dropout is False) & (do_l2 is False):
            self.reg = None

        with tfnn.name_scope('inputs'):
            self.data_placeholder = tfnn.placeholder(dtype=tfnn.float32,
                                                     shape=[None, self.input_size],
                                                     name='x_input')
            self.target_placeholder = tfnn.placeholder(dtype=tfnn.float32,
                                                       shape=[None, self.output_size],
                                                       name='y_input')
            if do_dropout:
                self.keep_prob_placeholder = tfnn.placeholder(dtype=tfnn.float32)
                tfnn.scalar_summary('dropout_keep_probability', self.keep_prob_placeholder)
                _reg_value = self.keep_prob_placeholder
            elif do_l2:
                self.l2_placeholder = tfnn.placeholder(tfnn.float32)
                tfnn.scalar_summary('l2_value', self.l2_placeholder)
                _reg_value = self.l2_placeholder
            else:
                _reg_value = None

        self.layers_configs = {
            'type': ['input'],
            'name': ['input_layer'],
            'neural_structure': [{'input_size': self.input_size, 'output_size': self.input_size}],
            'ntype': ntype,
        }
        self.layers_results = {
            'reg_value': _reg_value,
            'Layer': [None],
            'Wx_plus_b': [None],
            'activated': [None],
            'dropped': [None],
            'final': [self.data_placeholder]
        }

    def build_layers(self, layers):
        if isinstance(layers, Layer):
            layers.construct(self.layers_configs, self.layers_results)
            self._add_to_log(layers)
            if layers.layer_type == 'output':
                self._init_loss()
        elif isinstance(layers, (list, tuple)):
            for layer in layers:
                layer.construct(self.layers_configs, self.layers_results)
                self._add_to_log(layer)
                if layer.layer_type == 'output':
                    self._init_loss()
        else:
            raise ValueError('layers must be a list of layer objects, or a single layer object. '
                             'Not a %s' % type(layers))

    def add_hidden_layer(self, n_neurons, activator=None, dropout_layer=False,
                         w_initial='xavier', name=None,):
        """
        For original or simple neural network.
        """
        _layer = tfnn.HiddenLayer(n_neurons, activator, dropout_layer,
                                  w_initial, name)
        _layer.construct(self.layers_configs, self.layers_results)
        self._add_to_log(_layer)

    def add_fc_layer(self, n_neurons, activator=None, dropout_layer=False,
                     w_initial='xavier', name=None):
        _layer = tfnn.FCLayer(n_neurons, activator, dropout_layer,
                              w_initial, name)
        _layer.construct(self.layers_configs, self.layers_results)
        self._add_to_log(_layer)

    def add_conv_layer(self,
                       patch_x, patch_y, n_filters, activator=None,
                       strides=(1, 1), padding='SAME',
                       pooling='max', pool_strides=(2, 2), pool_k=(2, 2),
                       pool_padding='SAME', image_shape=None,
                       dropout_layer=False, w_initial='xavier', name=None,
                       ):
        _layer = tfnn.ConvLayer(
            patch_x, patch_y, n_filters, activator,
            strides, padding, pooling, pool_strides, pool_k,
            pool_padding, image_shape,
            dropout_layer, w_initial, name)
        _layer.construct(self.layers_configs, self.layers_results)
        self._add_to_log(_layer)

    def add_output_layer(self, activator=None, dropout_layer=False,
                         w_initial='xavier', name=None,):
        _layer = tfnn.OutputLayer(activator, dropout_layer,
                                  w_initial, name)
        _layer.construct(self.layers_configs, self.layers_results)
        self._add_to_log(_layer)
        self._init_loss()

    def set_learning_rate(self, lr, exp_decay=None):
        """

        :param lr:
        :param exp_decay: a dictionary like dict(decay_steps=None, decay_rate=None, staircase=False, name=None),
                        otherwise None.
        :return:
        """
        if isinstance(exp_decay, dict):
            if 'decay_steps' not in exp_decay:
                raise KeyError('Set decay_steps in exp_decay=dict(decay_steps)')
            if 'decay_rate' not in exp_decay:
                raise KeyError('Set decay_steps in exp_decay=dict(decay_rate)')
            if 'staircase' not in exp_decay:
                exp_decay['staircase'] = False
            if 'name' not in exp_decay:
                exp_decay['name'] = None
            self._lr = tfnn.train.exponential_decay(lr, self.global_step,
                                                   decay_steps=exp_decay['decay_steps'],
                                                   decay_rate=exp_decay['decay_rate'],
                                                   staircase=exp_decay['staircase'],
                                                   name=exp_decay['name'])
        else:
            self._lr = tfnn.constant(lr)
        tfnn.scalar_summary('learning_rate', self._lr)

    def set_optimizer(self, optimizer=None, *args, **kwargs):
        """

        :param optimizer: a string to represent the tensorflow optimizer name.
                        Available optimizers are list as below:
                        [
                        'gradient_descent',   or 'GD',
                        'adadelta',            or 'AD',
                        'adagrad',             or 'AG',
                        'momentum',            or 'MT',
                        'adam',                or 'Adam',
                        'ftrl',                or 'Ftrl',
                        'rmsprop',             or 'RMSProp'
                        ]

        :param args:
        :param kwargs:
        :return:
        """
        if optimizer is None:
            self._optimizer = tfnn.train.GradientDescentOptimizer
        elif optimizer.lower() in ['gradient_descent', 'gd', 'gradientdescent']:
            self._optimizer = tfnn.train.GradientDescentOptimizer
        elif optimizer.lower() in ['adadelta', 'ad']:
            self._optimizer = tfnn.train.AdadeltaOptimizer
        elif optimizer.lower() in ['adagrad', 'ag']:
            self._optimizer = tfnn.train.AdagradOptimizer
        elif optimizer.lower() in ['momentum', 'mt']:
            self._optimizer = tfnn.train.MomentumOptimizer
        elif optimizer.lower() in ['adam',]:
            self._optimizer = tfnn.train.AdamOptimizer
        elif optimizer.lower() in ['ftrl',]:
            self._optimizer = tfnn.train.FtrlOptimizer
        elif optimizer.lower() in ['rmsprop', ]:
            self._optimizer = tfnn.train.RMSPropOptimizer
        else:
            raise ValueError('''optimizer %s is not available, check the available optimizers:
                                [
                                'gradient_descent',   or 'GD',
                                'adadelta',            or 'AD',
                                'adagrad',             or 'AG',
                                'momentum',            or 'MT',
                                'adam',                or 'Adam',
                                'ftrl',                or 'Ftrl',
                                'rmsprop',             or 'RMSProp'
                                ]''' % optimizer)
        self.optimizer_params = [args, kwargs]
        if self.layers_configs['type'][-1] != 'output':
            raise NotImplementedError('Please add output layer.')

    def run_step(self, feed_xs, feed_ys, *args, **kwargs):
        if np.ndim(feed_xs) == 1:
            feed_xs = feed_xs[np.newaxis, :]
        if np.ndim(feed_ys) == 1:
            feed_ys = feed_ys[np.newaxis, :]
        self._check_init()
        _feed_dict = self._get_feed_dict(feed_xs, feed_ys, *args, **kwargs)
        self.sess.run(self._train_op, feed_dict=_feed_dict)

    def fit(self, feed_xs, feed_ys, steps=None, *args, **kwargs):
        def _print_log(log):
            print('\r{}'.format(log), end='')

        def _get_progress(t_cost, step, steps):
            _time_remaining_second = int(t_cost * (steps - step) / step)
            if _time_remaining_second > 60:
                _time_remaining_min = _time_remaining_second // 60
                _time_remaining_second %= 60
                if _time_remaining_min > 60:
                    _time_remaining_hour = _time_remaining_min // 60
                    _time_remaining_min %= 60
                    _time_remaining = str(_time_remaining_hour) + 'h-' \
                                      + str(_time_remaining_min) + 'm-' \
                                      + str(_time_remaining_second) + 's'
                else:
                    _time_remaining = str(_time_remaining_min) + 'm-' \
                                      + str(_time_remaining_second) + 's'
            else:
                _time_remaining = str(_time_remaining_second) + 's'

            _percentage = str(round(step / steps * 100, 2)) + '%'
            return [_time_remaining, _percentage]

        train_data = tfnn.Data(feed_xs, feed_ys)
        if steps is None:
            steps = train_data.n_samples
        time_start = time.time()
        for step in range(1, steps+1):
            b_xs, b_ys = train_data.next_batch(50)
            self.run_step(feed_xs=b_xs, feed_ys=b_ys, *args, **kwargs)
            if step % 200 == 0:
                time_cost = time.time() - time_start
                time_remaining, percentage = _get_progress(time_cost, step, steps)
                feed_dict = self._get_feed_dict(b_xs, b_ys, keep_prob=1., l2_value=0.)
                cost = self.sess.run(self.loss, feed_dict=feed_dict)
                _log = percentage + ' | ETA: ' + str(time_remaining) + ' | Cost: ' + \
                       str(round(cost, 5))
                _print_log(_log)
        print('\r')

    def predict(self, *args, **kwargs):
        raise NotImplementedError("Abstract method")

    def save(self, name='new_model', path=None, global_step=None, replace=False):
        if not hasattr(self, '_saver'):
            self._saver = tfnn.NetworkSaver()
        self._saver.save(self, name, path, global_step, replace=replace)

    def close(self):
        self.sess.close()

    def _add_to_log(self, layer):
        for key in layer.configs_dict.keys():
            self.layers_configs[key].append(layer.configs_dict[key])
        for key in layer.results_dict.keys():
            self.layers_results[key].append(layer.results_dict[key])

    def _check_init(self):
        if not hasattr(self, '_init'):
            if not hasattr(self, 'lr'):
                self.set_learning_rate(0.001)
            self.optimizer = self._optimizer(self._lr,  *self.optimizer_params[0], **self.optimizer_params[1])
            with tfnn.name_scope('trian'):
                self._train_op = self.optimizer.minimize(self.loss, self.global_step, name='train_op')
            # initialize all variables
            self._init = tfnn.initialize_all_variables()
            self.sess = tfnn.Session()
            self.sess.run(self._init)

    def _init_loss(self):
        """do not use in network.py"""
        self.loss = None

    def _get_feed_dict(self, xs, ys, *args, **kwargs):
        if self.reg == 'dropout':
            if args:
                kp = args[0]
            else:
                kp = kwargs['keep_prob']

            if not hasattr(self, '_keep_prob'):
                self._keep_prob = tfnn.constant(kp)

            _feed_dict = {
                self.data_placeholder: xs,
                self.target_placeholder: ys,
                self.keep_prob_placeholder: kp
            }
        elif self.reg == 'l2':
            if args:
                l2_value = args[0]
            else:
                l2_value = kwargs['l2_value']

            if not hasattr(self, '_l2_value'):
                self._l2_value = tfnn.constant(l2_value)

            _feed_dict = {
                self.data_placeholder: xs,
                self.target_placeholder: ys,
                self.l2_placeholder: l2_value
            }
        else:
            _feed_dict = {
                self.data_placeholder: xs,
                self.target_placeholder: ys
            }
        return _feed_dict

    def __add__(self, _layers):
        self.build_layers(_layers)
        return self

    def __iadd__(self, _layers):
        self.build_layers(_layers)
        return self

    def __len__(self):
        """ return the number of layers that are added to network """
        return len(self.layers_configs['type']) - 1

    @property
    def lr(self):
        return self._lr

    @property
    def keep_prob(self):
        return self._keep_prob

    @property
    def l2_value(self):
        return self._l2_value

    @property
    def Ws(self, n_layer=None):
        if not (n_layer is None or type(n_layer) is int):
            raise TypeError('layer must to be None or int')
        if n_layer is None:
            _Ws = []
            for layer in self.layers_results['Layer'][1:]:
                _W = self.sess.run(layer.W)
                _Ws.append(_W)
        else:
            if n_layer >= len(self.layers_results):
                raise IndexError('Do not have layer %i' % n_layer)
            _Ws = self.sess.run(self.layers_results['Layer'][1:][n_layer].W)
        return _Ws

    @property
    def Wshape(self, n_layer=None):
        if not (n_layer is None or type(n_layer) is int):
            raise TypeError('layer must to be None or int')
        if n_layer is None:
            _Wshape = []
            for layer in self.layers_results['Layer'][1:]:
                _Wshape.append(layer.W.get_shape())
        else:
            if n_layer >= len(self.layers_results):
                raise IndexError('Do not have layer %i' % n_layer)
            _Wshape = self.layers_results['Layer'][1:][n_layer].W.get_shape()
        return _Wshape

    @property
    def bs(self, n_layer=None):
        if not (n_layer is None or type(n_layer) is int):
            raise TypeError('layer need to be None or int')
        if n_layer is None:
            _bs = []
            for layer in self.layers_results['Layer'][1:]:
                _b = self.sess.run(layer.b)
                _bs.append(_b)
        else:
            if n_layer >= len(self.layers_results):
                raise IndexError('Do not have layer %i' % n_layer)
            _bs = self.sess.run(self.layers_results['Layer'][1:][n_layer].b)
        return _bs

    @property
    def bshape(self, n_layer=None):
        if not (n_layer is None or type(n_layer) is int):
            raise TypeError('layer must to be None or int')
        if n_layer is None:
            _bshape = []
            for layer in self.layers_results['Layer'][1:]:
                _bshape.append(layer.b.get_shape())
        else:
            if n_layer >= len(self.layers_results):
                raise IndexError('Do not have layer %i' % n_layer)
            _bshape = self.layers_results['Layer'][1:][n_layer].b.get_shape()
        return _bshape

