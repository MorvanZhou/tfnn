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
        self.n_neurons = layers_configs['net_in_out'].iloc[0]['output_size']
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
        if layers_configs['type'].iloc[-1] == 'conv':
            conv_shape = layers_configs['neural_structure'].iloc[-1]['output_size']
            flat_shape = conv_shape[0] * conv_shape[1] * conv_shape[2]
            layers_configs['neural_structure'].iloc[-1]['output_size'] = flat_shape

            flat_result = tfnn.reshape(
                layers_results['final'].iloc[-1],
                [-1, flat_shape], name='flat4fc')
            layers_results['final'].iloc[-1] = flat_result
        elif layers_configs['type'].iloc[-1] == 'fc':
            pass
        else:
            raise TypeError('The first Fully connected layer should followed by a Convolutional layer')

        self._construct(self.n_neurons, layers_configs, layers_results)
