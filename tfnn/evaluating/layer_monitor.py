import matplotlib.pyplot as plt
from tfnn.evaluating.monitor import Monitor


class LayerMonitor(Monitor):
    def __init__(self, grid_space, objects, evaluator):
        super(LayerMonitor, self).__init__(evaluator, 'layer_monitor')
        self._network = self.evaluator.network
        self._objects = objects
        self._axes = {}
        self._fig = plt.figure(figsize=(13, 13))
        neural_structure = self._network.layers_configs['neural_structure']

        # for input layer
        res_name = 'input'
        self._axes[res_name] = plt.subplot2grid(grid_space, (0, 1), )
        self._axes[res_name].tick_params(axis='both', which='both',
                                         bottom='off', top='off', left='off', right='off',
                                         labelbottom='off', labelleft='off', )
        self._axes[res_name].grid(False)
        self._axes[res_name].set_title(r'$%s$' % res_name)
        res_x_label = neural_structure.iloc[1]['input_size']
        self._axes[res_name].set_xlabel(r'$%i\ inputs$' % res_x_label)

        for r_loc, name in enumerate(objects):
            W_name = 'W_' + str(name+1)
            res_name = 'output_' + str(name+1)

            self._axes[W_name] = plt.subplot2grid(grid_space, (r_loc+1, 0),)
            self._axes[res_name] = plt.subplot2grid(grid_space, (r_loc+1, 1),)
            self._axes[W_name].tick_params(axis='both', which='both',
                                           bottom='off', top='off', left='off', right='off',
                                           labelbottom='off', labelleft='off',)
            self._axes[W_name].grid(False)
            self._axes[res_name].tick_params(axis='both', which='both',
                                             bottom='off', top='off', left='off', right='off',
                                             labelbottom='off', labelleft='off', )

            self._axes[res_name].grid(False)
            self._axes[W_name].set_title(r'$%s$' % W_name)
            self._axes[res_name].set_title(r'$%s$' % res_name)

            W_y_label = neural_structure.iloc[name+1]['input_size']
            W_x_label = neural_structure.iloc[name+1]['output_size']
            res_x_label = W_x_label
            self._axes[W_name].set_ylabel(r'$%i\ inputs$' % W_y_label)
            self._axes[W_name].set_xlabel(r'$%i\ outputs$' % W_x_label)
            self._axes[res_name].set_xlabel(r'$%i\ neurons$' % res_x_label)
        self._fig.subplots_adjust(hspace=0.3)
        self._cbar_ax = self._fig.add_axes([0.15, 0.05, 0.7, 0.02])
        plt.ion()
        plt.show()

    def monitoring(self, t_xs, t_ys):
        all_Ws = self._network.sess.run(list(self._network.Ws.values))
        feed_dict = self.evaluator.get_feed_dict(t_xs, t_ys)
        all_outputs = self._network.sess.run(list(self._network.layers_results['activated'].values)[1:],
                                             feed_dict=feed_dict)
        # for 1st layer
        res_name = 'input'
        self._axes[res_name].imshow(t_xs, interpolation='nearest', cmap='rainbow', origin='lower')
        res_y_label = len(all_outputs[0])
        self._axes[res_name].set_ylabel(r'$batch\ size:%i$' % res_y_label)
        for name in self._objects:
            W_name = 'W_' + str(name + 1)
            res_name = 'output_' + str(name + 1)
            image = self._axes[W_name].imshow(all_Ws[name], interpolation='nearest', cmap='rainbow', origin='lower')
            self._axes[res_name].imshow(all_outputs[name], interpolation='nearest', cmap='rainbow', origin='lower')
            res_y_label = len(all_outputs[name])
            self._axes[res_name].set_ylabel(r'$batch\ size:%i$' % res_y_label)

        self._fig.colorbar(image, cax=self._cbar_ax, orientation='horizontal')
        plt.pause(0.01)


    @staticmethod
    def hold_plot():
        plt.ioff()
        plt.show()

