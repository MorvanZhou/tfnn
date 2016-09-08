import tfnn
from tfnn.body.layer import Layer


class HiddenLayer(Layer):
    def __init__(self,
                 n_neurons, activator=None, dropout_layer=False,
                 w_initial='xavier', name=None,):
        super(HiddenLayer, self).__init__(activator, dropout_layer,
                                          w_initial, name,
                                          layer_type='hidden')
        self.n_neurons = n_neurons

    def construct(self, layers_configs, layers_results):
        self._construct(self.n_neurons, layers_configs, layers_results)


class OutputLayer(Layer):
    def __init__(self,
                 activator=None, dropout_layer=False,
                 w_initial='xavier', name=None,):
        super(OutputLayer, self).__init__(activator, dropout_layer,
                                          w_initial, name,
                                          layer_type='output')
        self.n_neurons = None

    def construct(self, layers_configs, layers_results):
        net_type = layers_configs[0]['para']['ntype']
        if (net_type == 'CNet') and (self.activator is not None):
            raise AttributeError('The activator in output layer for classification neural network has to be None')
        self.n_neurons = layers_configs[0]['net_in_out']['output_size']
        self._construct(self.n_neurons, layers_configs, layers_results)


class FCLayer(Layer):
    def __init__(self,
                 n_neurons, activator=None, dropout_layer=False,
                 w_initial='xavier', name=None, ):
        super(FCLayer, self).__init__(activator, dropout_layer,
                                      w_initial, name,
                                      layer_type='fc')
        self.n_neurons = n_neurons

    def construct(self, layers_configs, layers_results):
        if layers_configs[-1]['type'] == 'conv':
            conv_shape = layers_configs[-1]['neural_structure']['output_size']
            flat_shape = conv_shape[0] * conv_shape[1] * conv_shape[2]
            layers_configs[-1]['neural_structure']['output_size'] = flat_shape

            flat_result = tfnn.reshape(
                layers_results[-1]['final'],
                [-1, flat_shape], name='flat4fc')
            layers_results[-1]['final'] = flat_result
        elif layers_configs[-1]['type'] == 'fc':
            pass
        else:
            raise TypeError('The first Fully connected layer should followed by a Convolutional layer')

        self._construct(self.n_neurons, layers_configs, layers_results)
