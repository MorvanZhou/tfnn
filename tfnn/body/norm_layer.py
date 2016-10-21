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
        self._params['n_neurons'] = self.n_neurons

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
        net_type = layers_configs['ntype']
        if (net_type == 'CNet') and (self.activator is not None):
            raise AttributeError('The activator in output layer for classification neural network has to be None')
        self.n_neurons = layers_configs['params'][0]['output_size']
        self._construct(self.n_neurons, layers_configs, layers_results)


class FCLayer(Layer):
    def __init__(self,
                 n_neurons, activator=None, dropout_layer=False,
                 w_initial='xavier', name=None, ):
        super(FCLayer, self).__init__(activator, dropout_layer,
                                      w_initial, name,
                                      layer_type='fc')
        self.n_neurons = n_neurons
        self._params['n_neurons'] = self.n_neurons

    def construct(self, layers_configs, layers_results):
        if layers_configs['type'][-1] == 'conv':
            conv_shape = layers_configs['neural_structure'][-1]['output_size']
            flat_shape = conv_shape[0] * conv_shape[1] * conv_shape[2]
            layers_configs['neural_structure'][-1]['output_size'] = flat_shape

            flat_result = tfnn.reshape(
                layers_results['final'][-1],
                [-1, flat_shape], name='flat4fc')
            layers_results['final'][-1] = flat_result
        elif layers_configs['type'][-1] == 'fc':
            pass
        else:
            raise TypeError('The first Fully connected layer should followed by a Convolutional layer')

        self._construct(self.n_neurons, layers_configs, layers_results)
