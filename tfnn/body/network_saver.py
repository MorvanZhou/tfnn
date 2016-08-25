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
        if path[0] != '/':
            path = '/' + path
        if path[-1] != '/':
            path = path + '/'

        check_dir = os.getcwd() + path
        if os.path.isdir(check_dir):
            save_path = check_dir
        elif os.path.isdir(path):
            save_path = path
        else:
            raise NotADirectoryError('the directory is not exist: %s' % path)

        saver = tfnn.train.Saver()
        variables_path = save_path+'variables/'
        config_path = save_path+'configs/'
        if not os.path.exists(variables_path):
            os.makedirs(variables_path)
        if not os.path.exists(config_path):
            os.makedirs(config_path)
        model_path = saver.save(network.sess, variables_path + 'network.ckpt')
        network_configs = {'name': network.name,
                          'layers_configs': network.layers_configs,
                          'data_config': network.normalizer.config}
        with open(config_path+'network_configs.pickle', 'wb') as file:
            pickle.dump(network_configs, file)
        print("Model saved in file: %s" % save_path)
        self._network = network

    def restore(self, path='tmp/'):
        check_dir = os.getcwd() + path
        if os.path.isdir(check_dir):
            path = check_dir
        elif os.path.isdir(path):
            path = path
        else:
            raise NotADirectoryError('the directory is not exist: %s' % path)

        config_path = path + '/configs/network_configs.pickle'

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
                network.add_conv_layer(para['patch_x'], para['patch_y'], para['n_features'],
                                       activator, para['pool'], layer_drop, para['image_shape'],
                                       layer_name)
        network.sess = tfnn.Session()
        self._network = network
        saver = tfnn.train.Saver()
        self._network._init = tfnn.initialize_all_variables()
        self._network.sess.run(self._network._init)
        saver.restore(self._network.sess, path + '/variables/network.ckpt')
        print('Model restored.')
        return self._network

if __name__ == '__main__':
    from sklearn.datasets import load_boston
    from tensorflow.python.platform import gfile
    xs = load_boston().data
    ys = load_boston().target
    data = tfnn.Data(xs, ys)


    def save():
        network = tfnn.RegNetwork(xs.shape[1], ys.shape[1], do_dropout=True)
        network.add_hidden_layer(100, activator=tfnn.nn.relu, dropout_layer=True)
        network.add_output_layer(activator=None)
        optimizer = tfnn.train.GradientDescentOptimizer(0.5)
        network.set_optimizer(optimizer)
        tfnn.train.write_graph(network.sess.graph_def, '/tmp/load', name='test.pb', as_text=False)

        for i in range(1000):
            b_xs, b_ys = data.next_batch(50, loop=True)
            network.run_step(b_xs, b_ys, 0.5)
        saver = tfnn.train.Saver(tfnn.all_variables())
        saver.save(network.sess, "checkpoint.data")

    def restore():
        with tfnn.Session() as persisted_sess:
            with gfile.FastGFile("/tmp/load/test.pb", 'rb') as f:
                graph_def = tfnn.GraphDef()
                graph_def.ParseFromString(f.read())
                persisted_sess.graph.as_default()
                tfnn.import_graph_def(graph_def, name='')

    saver = NetworkSaver()
    network = saver.restore(path='/tmp/')
    print(network.layers_configs)
    network.layers_configs.to_csv()