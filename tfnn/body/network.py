import pandas as pd
import numpy as np
import tfnn
from tfnn.datasets.normalizer import Normalizer


class Network(object):
    def __init__(self, n_inputs, n_outputs, input_dtype, output_dtype, output_activator,
                 do_dropout, do_l2, seed=None):
        self.normalizer = Normalizer()
        self.input_size = n_inputs
        self.output_size = n_outputs
        self.input_dtype = input_dtype
        self.output_dtype = output_dtype
        self.output_activator = output_activator
        if do_dropout and do_l2:
            raise ValueError('Cannot do dropout and l2 at once. Choose only one of them.')
        if do_dropout:
            self.reg = 'dropout'
        if do_l2:
            self.reg = 'l2'
        if (do_dropout is False) & (do_l2 is False):
            self.reg = None
        self.seed = seed

        with tfnn.name_scope('inputs'):
            self.data_placeholder = tfnn.placeholder(dtype=input_dtype, shape=[None, n_inputs], name='x_input')
            self.target_placeholder = tfnn.placeholder(dtype=output_dtype, shape=[None, n_outputs], name='y_input')
            if do_dropout:
                self.keep_prob_placeholder = tfnn.placeholder(dtype=tfnn.float32)
                tfnn.scalar_summary('dropout_keep_probability', self.keep_prob_placeholder)
            if do_l2:
                self.l2_placeholder = tfnn.placeholder(tfnn.float32)
                tfnn.scalar_summary('l2_lambda', self.l2_placeholder)
        _input_layer_configs = \
            {'type': ['input'],
             'name': ['input_layer'],
             'neural_structure': [{'input_size': n_inputs, 'output_size': n_inputs}],
             'para': [None]}
        _input_layer_results = \
            {'Wx_plus_b': [None],
             'activated': [None],
             'dropped': [None],
             'final': [self.data_placeholder]}

        self.layers_configs = pd.DataFrame(_input_layer_configs)
        self.layers_results = pd.DataFrame(_input_layer_results)
        self.Ws = pd.Series([])
        self.bs = pd.Series([])

    def add_fc_layer(self, n_neurons, activator=None, dropout_layer=False, name=None,
                     w_initial='xavier'):
        if self.layers_configs['type'].iloc[-1] == 'conv':
            conv_shape = self.layers_configs['neural_structure'].iloc[-1]['output_size']
            flat_shape = conv_shape[0] * conv_shape[1] * conv_shape[2]
            self.layers_configs['neural_structure'].iloc[-1]['output_size'] = flat_shape

            flat_result = tfnn.reshape(
                self.layers_results['final'].iloc[-1],
                [-1, flat_shape], name='flat4fc')
            self.layers_results['final'].iloc[-1] = flat_result

        self._add_layer(n_neurons, activator, dropout_layer, name, w_initial,
                        layer_type='fc')

    def add_hidden_layer(self, n_neurons, activator=None, dropout_layer=False, name=None,
                         w_initial='xavier'):
        """
        For original or simple neural network.
        :param n_neurons:
        :param activator:
        :param dropout_layer:
        :param name:
        :return:
        """
        self._add_layer(n_neurons, activator, dropout_layer, name, w_initial,
                        layer_type='hidden',)

    def add_conv_layer(self, patch_x, patch_y, n_filters, activator=None,
                       strd=(1, 1), pool='max', pool_strd=(2, 2), pool_k=(2, 2),
                       dropout_layer=False, image_shape=None, name=None, w_initial='xavier'):
        """

        :param patch_x:
        :param patch_y:
        :param n_features:
        :param activator: string or tfnn.nn.some_activation
        :param pool:
        :param dropout_layer:
        :param image_shape: should be a tuple or list of image (length, width, channels)
        :param name:
        :return:
        """
        def conv2d(image, filters, strd=(1, 1)):
            # stride [1, x_movement, y_movement, 1]
            # Must have strides[0] = strides[3] = 1
            return tfnn.nn.conv2d(input=image, filter=filters,
                                  strides=[1, strd[0], strd[1], 1], padding='SAME')

        def pool_2x2(image, layer_size, method='max', strd_x=2, strd_y=2, k_x=2, k_y=2):
            # stride [1, x_movement, y_movement, 1]
            if method == 'max':
                result = tfnn.nn.max_pool(value=image, ksize=[1, k_x, k_y, 1],
                                          strides=[1, strd_x, strd_y, 1], padding='SAME')
            elif method == 'average':
                result = tfnn.nn.avg_pool(value=image, ksize=[1, k_x, k_y, 1],
                                          strides=[1, strd_x, strd_y, 1], padding='SAME')
            else:
                raise ValueError('No method called %s' % method)
            length = layer_size[0]/strd_x
            width = layer_size[1]/strd_y
            features = layer_size[2]
            if not (type(length) == int) and (type(width) == int):
                raise ValueError('pooling dimension error')
            else:
                _out_size = [int(length), int(width), features]
            return [result, _out_size]

        def check_name(name):
            if name is None:
                layer_name = 'conv_layer'
            else:
                layer_name = name

            # check repeated name
            if self.layers_configs['name'].isin([layer_name]).any():
                _counter = 0
                while True:
                    _counter += 1
                    new_layer_name = layer_name + '_%i' % _counter
                    if not self.layers_configs['name'].isin([new_layer_name]).any():
                        layer_name = new_layer_name
                        break
            return layer_name

        def check_image_shape(image_shape):
            if image_shape is not None:
                if len(self.layers_configs) == 1:
                    if type(image_shape) is tuple:
                        image_shape = list(image_shape)
                    elif type(image_shape) is list:
                        pass
                    else:
                        raise ValueError('image_shape can only be a tuple or list')
                    self.layers_configs['neural_structure'].iloc[-1]['output_size'] = image_shape
                    _xs_placeholder = self.layers_results['final'].iloc[-1]
                    replaced_image_shape = image_shape.copy()
                    replaced_image_shape.insert(0, -1)
                    with tfnn.name_scope('reshape_inputs'):
                        _image_placeholder = tfnn.reshape(_xs_placeholder, replaced_image_shape)
                    self.layers_results['final'].iloc[-1] = _image_placeholder
                else:
                    raise IndexError('This is not the first conv layer, must not use image_shape')

        check_image_shape(image_shape)
        layer_name = check_name(name)
        # in conv, the _in_size should be the [length, width, channels]
        _in_size = self.layers_configs['neural_structure'].iloc[-1]['output_size']
        with tfnn.variable_scope(layer_name):
            with tfnn.variable_scope('weights') as weights_scope:
                W_conv = self._weight_variable(
                    [patch_x,       # patch length
                     patch_y,       # patch width
                     _in_size[-1],  # filter height / channels
                     n_filters],
                    w_initial)    # number of filters
                tfnn.histogram_summary(layer_name + '/weights', W_conv)

                # the image summary for visualizing filters
                weights_scope.reuse_variables()
                weights = tfnn.get_variable('weights', trainable=False)
                # scale weights to [0 255] and convert to uint8 (maybe change scaling?)
                x_min = tfnn.reduce_min(weights)
                x_max = tfnn.reduce_max(weights)
                weights_0_to_1 = (weights - x_min) / (x_max - x_min)
                weights_0_to_255_uint8 = tfnn.image.convert_image_dtype(weights_0_to_1, dtype=tfnn.uint8)
                # to tf.image_summary format [batch_size, height, width, channels]
                W_transposed = tfnn.transpose(weights_0_to_255_uint8, [3, 0, 1, 2])
                # image Tensor must be 4-D with last dim 1, 3, or 4,
                # (n_filter, length, width, channel)
                channels_to_look = 3
                if W_transposed._shape[-1] > channels_to_look:
                    n_chunks = int(W_transposed._shape[-1] // channels_to_look)
                    W_transposed = tfnn.split(3, n_chunks,
                                              W_transposed[:, :, :, :n_chunks*channels_to_look])[0]
                # this will display random 5 filters from the n_filters in conv
                tfnn.image_summary(layer_name + '/filters',
                                   W_transposed, max_images=10)

            with tfnn.variable_scope('biases'):
                b_conv = self._bias_variable([n_filters, ])
                tfnn.histogram_summary(layer_name + '/biases', b_conv)

            with tfnn.name_scope('Wx_plus_b'):
                product = conv2d(self.layers_results['final'].iloc[-1],
                                 W_conv, strd) + b_conv

            if activator is not None:
                if isinstance(activator, str):
                    activator = self._get_activator(activator)
                activated_product = activator(product)  # output size 28x28x32
            else:
                activated_product = product
            tfnn.histogram_summary(layer_name + '/activated_product', activated_product)

            _out_size = _in_size.copy()
            _out_size[-1] = n_filters
        # pooling process
        with tfnn.name_scope('pool'):
            pooled_product, _out_size = pool_2x2(activated_product, _out_size, method=pool,
                                                 strd_x=pool_strd[0], strd_y=pool_strd[1],
                                                 k_x=pool_k[0], k_y=pool_k[1])
            tfnn.histogram_summary(layer_name + '/pooled_product', pooled_product)

        if (self.reg == 'dropout') and dropout_layer:
            dropped_product = tfnn.nn.dropout(pooled_product,
                                              self.keep_prob_placeholder,
                                              seed=self.seed, name='dropout')
            final_product = dropped_product/self.keep_prob_placeholder
        else:
            dropped_product = None
            final_product = pooled_product

        activator_name = activated_product.name.split('/')[-1].split(':')[0] \
            if 'Wx_add_b' not in activated_product.name else None
        _layer_configs_dict = \
            {'type': 'conv',
             'name': layer_name,
             'neural_structure': {'input_size': _in_size, 'output_size': _out_size},
             'para': {'patch_x': patch_x, 'patch_y': patch_y, 'n_filters': n_filters,
                      'activator': activator_name, 'pool': pool, 'dropout_layer': dropout_layer,
                      'image_shape': image_shape, 'name': name, 'w_initial': w_initial}}
        _layer_results_dict = \
            {'Wx_plus_b': product,
             'activated': activated_product,
             'dropped': dropped_product,
             'final': final_product}

        self.layers_configs = self.layers_configs.append(_layer_configs_dict, ignore_index=True)
        self.layers_results = self.layers_results.append(_layer_results_dict, ignore_index=True)
        self.Ws.set_value(label=len(self.Ws), value=W_conv)
        self.bs.set_value(label=len(self.bs), value=b_conv)

    def add_output_layer(self, activator=None, dropout_layer=False, name=None,
                         w_initial='xavier'):
        self._add_layer(self.output_size, activator, dropout_layer, name,
                        w_initial, layer_type='output')
        self._init_loss()

    def set_optimizer(self, optimizer=None, global_step=None,):
        if optimizer is None:
            self._lr = 0.001
            optimizer = tfnn.train.GradientDescentOptimizer(self._lr)
        if self.layers_configs['type'].iloc[-1] != 'output':
            raise NotImplementedError('Please add output layer.')
        with tfnn.name_scope('trian'):
            if hasattr(optimizer, '_lr'):
                self._lr = optimizer._lr
            elif hasattr(optimizer, '_learning_rate'):
                self._lr = optimizer._learning_rate
            else:
                raise AttributeError('this optimizer %s dose not have _lr ot _learning rate'
                                     % optimizer._name)
            tfnn.scalar_summary('learning_rate', self._lr)
            self._train_op = optimizer.minimize(self.loss, global_step, name='train_op')
        self.sess = tfnn.Session()

    def run_step(self, feed_xs, feed_ys, keep_prob=None, l2=None):
        if np.ndim(feed_xs) == 1:
            feed_xs = feed_xs[np.newaxis, :]
        if np.ndim(feed_ys) == 1:
            feed_ys = feed_ys[np.newaxis, :]
        if not hasattr(self, '_init'):
            # initialize all variables
            self._init = tfnn.initialize_all_variables()
            self.sess.run(self._init)

        if self.reg == 'dropout':
            if keep_prob is None:
                raise ValueError('need pass a keep_prob for run_step')
            self.sess.run(self._train_op, feed_dict={
                self.data_placeholder: feed_xs,
                self.target_placeholder: feed_ys,
                self.keep_prob_placeholder: keep_prob})
        elif self.reg == 'l2':
            if l2 is None:
                raise ValueError('need pass a l2 for run_step')
            self.sess.run(self._train_op, feed_dict={
                self.data_placeholder: feed_xs,
                self.target_placeholder: feed_ys,
                self.l2_placeholder: l2})
        else:
            self.sess.run(self._train_op, feed_dict={
                self.data_placeholder: feed_xs,
                self.target_placeholder: feed_ys})

    def fit(self, feed_xs, feed_ys, steps=2000, *args):
        """
        Fit data to network, automatically training the network.
        :param feed_xs:
        :param feed_ys:
        :param steps: when n_iter=-1, the training steps= n_samples*2
        :param args: pass keep_prob when use dropout, pass l2_lambda when use l2 regularization.
        :return: Nothing
        """
        train_data = tfnn.Data(feed_xs, feed_ys)
        for _ in range(steps):
            b_xs, b_ys = train_data.next_batch(100, loop=True)
            self.run_step(feed_xs=b_xs, feed_ys=b_ys, *args)

    def get_loss(self, xs, ys):
        if self.reg == 'dropout':
            _loss_value = self.sess.run(self.loss, feed_dict={self.data_placeholder: xs,
                                                              self.target_placeholder: ys,
                                                              self.keep_prob_placeholder: 1.})
        elif self.reg == 'l2':
            _loss_value = self.sess.run(self.loss, feed_dict={self.data_placeholder: xs,
                                                              self.target_placeholder: ys,
                                                              self.l2_placeholder: 0})
        else:
            _loss_value = self.sess.run(self.loss, feed_dict={self.data_placeholder: xs,
                                                              self.target_placeholder: ys})
        return _loss_value

    def get_weights(self, layer=None):
        if not(layer is None or type(layer) is int):
            raise TypeError('layer need to be None or int')
        if layer is None:
            _Ws = []
            for W_layer in self.Ws:
                _W = self.sess.run(W_layer)
                _Ws.append(_W)
        else:
            if layer >= len(self.Ws):
                raise IndexError('Do not have layer %i' % layer)
            _Ws = self.sess.run(self.Ws[layer])
        return _Ws

    def get_biases(self, layer=None):
        if not(layer is None or type(layer) is int):
            raise TypeError('layer need to be None or int')
        if layer is None:
            _bs = []
            for b_layer in self.bs:
                _b = self.sess.run(b_layer)
                _bs.append(_b)
        else:
            if layer >= len(self.bs):
                raise IndexError('Do not have layer %i' % layer)
            _bs = self.sess.run(self.bs[layer])
        return _bs

    def predict(self, xs):
        pass

    def save(self, name='new_model', path=None, global_step=None, replace=False):
        if not hasattr(self, 'saver'):
            self._saver = tfnn.NetworkSaver()
        self._saver.save(self, name, path, global_step, replace=replace)

    def close(self):
        self.sess.close()

    def _add_layer(self, n_neurons, activator=None, dropout_layer=False, name=None,
                   w_initial='xavier', layer_type=None, ):
        """
        W shape(n_last_layer_neurons, n_this_layer_neurons]
        b shape(n_this_layer_neurons, ]
        product = tfnn.matmul(x, W) + b
        :param n_neurons: Number of neurons in this layer
        :param activator: The activation function
        :return:
        """
        def check_name(layer_type, name):
            if layer_type == 'hidden':
                if name is None:
                    layer_name = 'hidden_layer'
                else:
                    layer_name = name
            elif layer_type == 'output':
                if name is None:
                    layer_name = 'output_layer'
                else:
                    layer_name = name
            elif layer_type == 'fc':
                if name is None:
                    layer_name = 'fc_layer'
                else:
                    layer_name = name
            elif layer_type == 'conv':
                if name is None:
                    layer_name = 'conv_layer'
                else:
                    layer_name = name
            else:
                raise ValueError('layer_type not support %s' % layer_type)

            # check repeated name
            if self.layers_configs['name'].isin([layer_name]).any():
                _counter = 0
                while True:
                    _counter += 1
                    new_layer_name = layer_name + '_%i' % _counter
                    if not self.layers_configs['name'].isin([new_layer_name]).any():
                        layer_name = new_layer_name
                        break
            return layer_name

        layer_name = check_name(layer_type, name)

        _input_size = self.layers_configs['neural_structure'].iloc[-1]['output_size']   # this is from last layer
        with tfnn.variable_scope(layer_name):

            with tfnn.variable_scope('weights') as weights_scope:
                W = self._weight_variable([_input_size, n_neurons], initialize=w_initial)
                tfnn.histogram_summary(layer_name + '/weights', W)

                # the image summary for visualizing filters
                weights_scope.reuse_variables()
                # weights shape [n_inputs, n_hidden_units]
                weights = tfnn.get_variable('weights', trainable=False)
                # scale weights to [0 255] and convert to uint8 (maybe change scaling?)
                x_min = tfnn.reduce_min(weights)
                x_max = tfnn.reduce_max(weights)
                weights_0_to_1 = (weights - x_min) / (x_max - x_min)
                weights_0_to_255_uint8 = tfnn.image.convert_image_dtype(weights_0_to_1, dtype=tfnn.uint8)
                # to tf.image_summary format [batch_size, height, width, channels]
                # (1, n_neurons, weights, 1)
                W_expanded = tfnn.expand_dims(
                    tfnn.expand_dims(weights_0_to_255_uint8, 0), 3)
                tfnn.image_summary(layer_name + '/weights', W_expanded)

            with tfnn.variable_scope('biases'):
                b = self._bias_variable([n_neurons, ])
                tfnn.histogram_summary(layer_name + '/biases', b)

            with tfnn.name_scope('Wx_plus_b'):
                product = tfnn.add(tfnn.matmul(self.layers_results['final'].iloc[-1], W, name='Wx'),
                                   b, name='Wx_add_b')

            if activator is None:
                activated_product = product
            else:
                if isinstance(activator, str):
                    activator = self._get_activator(activator)
                activated_product = activator(product)
            tfnn.histogram_summary(layer_name+'/activated_product', activated_product)

            if (self.reg == 'dropout') and dropout_layer:
                dropped_product = tfnn.nn.dropout(activated_product,
                                                  self.keep_prob_placeholder,
                                                  seed=self.seed, name='dropout')
                final_product = dropped_product/self.keep_prob_placeholder
            else:
                dropped_product = None
                final_product = activated_product

        activator_name = activated_product.name.split('/')[-1].split(':')[0] \
            if 'Wx_add_b' not in activated_product.name else None
        _layer_configs_dict = \
            {'type': layer_type,
             'name': layer_name,
             'neural_structure': {'input_size': _input_size, 'output_size': n_neurons},
             'para': {'n_neurons': n_neurons, 'activator': activator_name,
                      'dropout_layer': dropout_layer, 'name': name, 'w_initial': w_initial}}
        _layer_results_dict = \
            {'Wx_plus_b': product,
             'activated': activated_product,
             'dropped': dropped_product,
             'final': final_product}
        self.layers_configs = self.layers_configs.append(_layer_configs_dict, ignore_index=True)
        self.layers_results = self.layers_results.append(_layer_results_dict, ignore_index=True)
        self.Ws.set_value(label=len(self.Ws), value=W)
        self.bs.set_value(label=len(self.bs), value=b)

    def _weight_variable(self, shape, initialize='xavier', name='weights'):
        # using a standard deviation of 1/sqrt(N), where N is the number of inputs to the given neuron layer.
        # stddev=1./np.sqrt(shape[0])
        # for relu the stddev = 1./np.sqrt(shape[0] / 2)
        if initialize == 'truncated_normal':
            initializer = tfnn.truncated_normal_initializer(mean=0., stddev=1./np.sqrt(shape[0]/2))
        elif initialize == 'random_normal':
            initializer = tfnn.random_normal_initializer(mean=0., stddev=1./np.sqrt(shape[0]/2))
        elif initialize == 'xavier':
            # uniform = True for using uniform distribution being the range: x = sqrt(6. / (in + out))
            # uniform = False for using normal distribution with a standard deviation of sqrt(3. / (in + out))
            initializer = tfnn.contrib.layers.xavier_initializer(uniform=False)
        else:
            raise ValueError('initializer not support %s' % initialize)
        return tfnn.get_variable(name=name, shape=shape, dtype=self.input_dtype,
                                 initializer=initializer, trainable=True,)

    def _bias_variable(self, shape, name='biases'):
        return tfnn.get_variable(name=name, shape=shape, dtype=self.input_dtype,
                                 initializer=tfnn.constant_initializer(0.1),
                                 trainable=True)

    def _init_loss(self):
        """do not use in network.py"""
        self.loss = None

    @staticmethod
    def _get_activator(name):
        if name == 'relu':
            activator = tfnn.nn.relu
        elif name == 'relu6':
            activator = tfnn.nn.relu6
        elif name == 'tanh':
            activator = tfnn.nn.tanh
        elif name == 'sigmoid':
            activator = tfnn.nn.sigmoid
        elif name == 'elu':
            activator = tfnn.nn.elu
        elif name == 'softplus':
            activator = tfnn.nn.softplus
        elif name == 'softsign':
            activator = tfnn.nn.softsign
        elif name == 'softmax':
            activator = tfnn.nn.softmax
        else:
            raise ValueError(
                '''the activation function %s is not supported. Function available:
                [relu, relu6, elu, tanh, sigmoid, softplus, softsign, softmax]''' % name)
        return activator
