import pickle
import tfnn
import os


class NetworkSaver(object):
    """
    Save, rebuild and restore network.
    """
    def save(self, network, path='/tmp/'):
        """
        save network config and normalized data config
        :param network: trained network
        :param path: save to path
        """
        saver = tfnn.train.Saver()
        save_path = os.getcwd() + path
        variables_path = save_path+'variables/'
        config_path = save_path+'config/'
        if not os.path.exists(variables_path):
            os.makedirs(variables_path)
        if not os.path.exists(config_path):
            os.makedirs(config_path)
        model_path = saver.save(network.sess, variables_path + 'network.ckpt')
        network_config = {'name': network.name,
                          'n_inputs': network.n_inputs,
                          'n_outputs': network.n_outputs,
                          'layers': [network.record_neurons, network.record_activators],
                          'data_config': network.normalizer.config}
        with open(config_path+'network_config.pickle', 'wb') as file:
            pickle.dump(network_config, file)
        print("Model saved in file: %s" % save_path)
        self._network = network

    def restore(self, path='/tmp/'):
        config_path = os.getcwd() + path + '/config/network_config.pickle'
        with open(config_path, 'rb') as file:
            network_config = pickle.load(file)
        name = network_config['name']  # network.name,
        n_inputs = network_config['n_inputs']  # network.n_inputs,
        n_outputs = network_config['n_outputs']  # network.n_outputs,
        layers_neurons = network_config['layers'][0]  # [network.record_neurons,
        layers_activators = network_config['layers'][1]  # network.record_activators],
        data_config = network_config['data_config']  # network.normalizer.config
        if name == 'RegressionNetwork':
            network = tfnn.RegNetwork(n_inputs, n_outputs)
        else:
            network = tfnn.ClfNetwork(n_inputs, n_outputs)
        network.normalizer.set_config(data_config)
        for i in range(len(layers_neurons)):
            layer_activator = layers_activators[i]
            if layer_activator is None:
                activator = None
            elif 'Relu:' in layer_activator:
                activator = tfnn.nn.relu
            elif 'Relu6:' in layer_activator:
                activator = tfnn.nn.relu6
            elif 'Softplus:' in layer_activator:
                activator = tfnn.nn.softplus
            elif 'Sigmoid:' in layer_activator:
                activator = tfnn.sigmoid
            elif 'Tanh:' in layer_activator:
                activator = tfnn.tanh
            else:
                raise ValueError('No activator as %s.' % layer_activator)
            if i != len(layers_neurons) - 1:
                network.add_hidden_layer(layers_neurons[i], activator)
            else:
                network.add_output_layer(activator)
            network.sess = tfnn.Session()
        self._network = network
        saver = tfnn.train.Saver()
        self._network._init = tfnn.initialize_all_variables()
        self._network.sess.run(self._network._init)
        saver.restore(self._network.sess, os.getcwd() + path + '/variables/network.ckpt')
        print('Model restored.')
        return self._network
