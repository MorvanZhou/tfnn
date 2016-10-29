import tfnn
import numpy as np


class Layer(object):
    def __init__(self, activator, dropout_layer, w_initial, name,
                 layer_type,):
        """

        :param activator: string or tfnn.nn.an_activation
        :param dropout_layer:
        :param w_initial:
        :param name:
        :param layer_type:
        """
        self._check_activator(activator)
        self.dropout_layer = dropout_layer
        self.w_initial = w_initial
        self.name = name
        self.layer_type = layer_type
        self._params = {
                 'activator': self.activator_name,
                 'dropout_layer': self.dropout_layer,
                 'name': self.name,
                 'w_initial': self.w_initial
             }

    def __str__(self):
        return self.name

    def construct(self, *args, **kwargs):
        raise NotImplementedError("Abstract method")

    def get_Wshape(self):
        return self.W.get_shape()

    def get_bshape(self):
        return self.b.get_shape()

    def _construct(self, n_neurons, layers_configs, layers_results):
        self.name = self._check_name(layers_configs)
        _input_size = layers_configs['neural_structure'][-1]['output_size']  # this is from last layer
        with tfnn.variable_scope(self.name):

            with tfnn.variable_scope('weights') as weights_scope:
                self.W = self._weight_variable([_input_size, n_neurons], initialize=self.w_initial)
                tfnn.histogram_summary(self.name + '/weights', self.W)

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
                tfnn.image_summary(self.name + '/weights', W_expanded)

            with tfnn.variable_scope('biases'):
                self.b = self._bias_variable([n_neurons, ])
                tfnn.histogram_summary(self.name + '/biases', self.b)

            with tfnn.name_scope('Wx_plus_b'):
                product = tfnn.add(tfnn.matmul(layers_results['final'][-1], self.W, name='Wx'),
                                   self.b, name='Wx_add_b')

            if self.activator is None:
                activated_product = product
            else:
                activated_product = self.activator(product)
            tfnn.histogram_summary(self.name + '/activated_product', activated_product)

            _do_dropout = layers_configs['params'][0]['do_dropout']
            if _do_dropout and self.dropout_layer:
                _keep_prob = layers_results['reg_value']
                dropped_product = tfnn.nn.dropout(activated_product,
                                                  _keep_prob,
                                                  name='dropout')
                final_product = dropped_product         # don't have to rescale it back, tf dropout has done this
            else:
                dropped_product = None
                final_product = activated_product

        self.configs_dict = {
            'type': self.layer_type,
            'name': self.name,
            'neural_structure': {'input_size': _input_size, 'output_size': n_neurons},
            'params': self._params,
        }
        self.results_dict = {
            'Layer': self,
            'Wx_plus_b': product,
            'activated': activated_product,
            'dropped': dropped_product,
            'final': final_product
        }

    def _check_name(self, layers_configs):
        if self.name is None:
            if self.layer_type == 'hidden':
                layer_name = 'hidden_layer'
            elif self.layer_type == 'output':
                layer_name = 'output_layer'
            elif self.layer_type == 'fc':
                layer_name = 'fc_layer'
            elif self.layer_type == 'conv':
                layer_name = 'conv_layer'
            else:
                raise ValueError('layer_type not support %s' % self.layer_type)
        else:
            layer_name = self.name

        # check repeated name
        layer_name_copy = layer_name
        _counter = 0
        while layer_name_copy in layers_configs['name']:
            _counter += 1
            layer_name_copy = layer_name + '_%i' % _counter
            if layer_name_copy not in layers_configs['name']:
                layer_name = layer_name_copy
                break
        return layer_name

    @staticmethod
    def _weight_variable(shape, initialize='xavier', name='weights'):
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
            raise ValueError('''initializer not support %s. The available initializers include:
            ['truncated_normal', 'random_normal', 'xavier']''' % initialize)
        return tfnn.get_variable(name=name, shape=shape, dtype=tfnn.float32,
                                 initializer=initializer, trainable=True,)

    @staticmethod
    def _bias_variable(shape, name='biases'):
        return tfnn.get_variable(name=name, shape=shape, dtype=tfnn.float32,
                                 initializer=tfnn.constant_initializer(1./np.sqrt(shape[0])),
                                 trainable=True)

    @staticmethod
    def _get_activator(name):
        name = name.lower()
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

    def _check_activator(self, activator):
        if isinstance(activator, str):
            self.activator_name = activator
            self.activator = self._get_activator(self.activator_name)
        elif isinstance(activator, type(None)):
            self.activator_name = activator
            self.activator = activator
        else:
            self.activator_name = activator(0).op.type.lower()
            self.activator = activator