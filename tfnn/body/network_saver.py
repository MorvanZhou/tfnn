import pickle
import tfnn
import os


class NetworkSaver(object):
    """
    Save, rebuild and restore network.
    """
    def __init__(self):
        self._configs_saved = False
        self._available_checkpoints = []

    def save(self, network, name='new_model', path=None, global_step=None, replace=False):
        """
        save network config and normalized data config
        :param network: trained network
        :param path: save to path
        :param global_step: default None.
        """
        self._network = network

        if path is None:
            path = '/'
        if path[0] != '/':
            path = '/' + path
        if path[-1] != '/':
            path += '/'

        check_dir = os.getcwd() + path
        if os.path.isdir(check_dir):
            save_path = check_dir
        elif os.path.isdir(path):
            save_path = path
        else:
            raise NotADirectoryError('the directory is not exist: %s' % path)

        if os.path.isdir(save_path+name) and (not self._configs_saved):
            if replace:
                save_path = save_path + name
            else:
                replace_answer = input('%s in %s already exists, replace it? (y/n)' % (name, save_path))
                if replace_answer == 'y':
                    save_path = save_path + name
                else:
                    raise FileExistsError('%s in %s already exists' % (name, save_path))
        else:
            save_path = save_path + name
        _saver = tfnn.train.Saver()

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        model_path = _saver.save(network.sess, save_path + '/net_variables',
                                global_step=global_step, write_meta_graph=False)

        if global_step is not None:
            self._available_checkpoints.append(global_step)
            with open(save_path + '/available_cps.pickle', 'wb') as file:
                pickle.dump(self._available_checkpoints, file)

        if not self._configs_saved:
            self._configs_saved = True
            network_configs = {'name': network.name,
                              'layers_configs': network.layers_configs,
                              'data_config': network.normalizer.config}
            with open(save_path+'/net_configs.pickle', 'wb') as file:
                pickle.dump(network_configs, file)

    def restore(self, name='new_model', path=None, checkpoint=None):
        if path is None:
            path = '/'
        if path[0] != '/':
            path = '/' + path
        if path[-1] != '/':
            path += '/'

        check_dir = os.getcwd() + path
        if os.path.isdir(check_dir):
            path = check_dir
        elif os.path.isdir(path):
            path = path
        else:
            raise NotADirectoryError('the directory is not exist: %s' % path)

        config_path = path + name + '/net_configs.pickle'

        # reset graph to build a new network independently
        tfnn.reset_default_graph()

        with open(config_path, 'rb') as file:
            network_config = pickle.load(file)
        net_name = network_config['name']  # network.name,
        layers_configs = network_config['layers_configs']  # network.n_inputs,
        data_config = network_config['data_config']  # network.normalizer.config
        input_size, output_size = layers_configs['neural_structure'].iloc[0]['input_size'], \
                                  layers_configs['neural_structure'].iloc[-1]['output_size']
        # select the type of network
        if net_name == 'RegressionNetwork':
            network = tfnn.RegNetwork(input_size, output_size)
        else:
            network = tfnn.ClfNetwork(input_size, output_size)
        # set the data configuration
        if data_config is not None:
            network.normalizer.set_config(data_config)
        # set each layer
        for index, layer_configs in layers_configs.iterrows():
            if index == 0:
                continue
            para = layer_configs['para']
            layer_name = para['name']
            layer_activator = para['activator']
            layer_drop = para['dropout_layer']
            layer_type = layer_configs['type']

            if layer_activator is None:
                activator = None
            elif layer_activator == 'Relu6':
                activator = tfnn.nn.relu6
            elif layer_activator == 'Relu':
                activator = tfnn.nn.relu
            elif layer_activator == 'Softplus':
                activator = tfnn.nn.softplus
            elif layer_activator == 'Sigmoid':
                activator = tfnn.sigmoid
            elif layer_activator == 'Tanh':
                activator = tfnn.tanh
            else:
                raise ValueError('No activator as %s.' % layer_activator)
            if layer_type == 'hidden':
                network.add_hidden_layer(para['n_neurons'], activator, layer_drop, layer_name)
            elif layer_type == 'fc':
                network.add_fc_layer(para['n_neurons'], activator, layer_drop, layer_name)
            elif layer_type == 'output':
                network.add_output_layer(activator, layer_drop, layer_name)
            elif layer_type == 'conv':
                network.add_conv_layer(para['patch_x'], para['patch_y'], para['n_filters'],
                                       activator, para['pool'], layer_drop, para['image_shape'],
                                       layer_name)
        network.sess = tfnn.Session()
        self._network = network
        _saver = tfnn.train.Saver()
        self._network._init = tfnn.initialize_all_variables()
        self._network.sess.run(self._network._init)
        if checkpoint is not None:
            var_path = '/net_variables-%i' % checkpoint
        else:
            var_path = '/net_variables'
        try:
            _saver.restore(self._network.sess, path + name + var_path)
        except ValueError:
            with open(path + name + '/available_cps.pickle', 'rb') as file:
                available_checkpoints = pickle.load(file)
            raise ValueError('Please define a checkpoint value for restoring. The available checkpoints are:',
                             available_checkpoints)

        return self._network
