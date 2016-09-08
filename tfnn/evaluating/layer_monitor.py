import matplotlib.pyplot as plt
import matplotlib as mpl
from tfnn.evaluating.monitor import Monitor
import numpy as np


class LayerMonitor(Monitor):
    def __init__(self, grid_space, objects, evaluator, figsize=(13, 13), cbar_range=(-1, 1), cmap='rainbow',
                 sleep=0.001):
        super(LayerMonitor, self).__init__(evaluator, 'layer_monitor')
        self._network = self.evaluator.network
        self._objects = objects
        self._cbar_range = cbar_range
        self._fig = plt.figure(figsize=figsize)
        self._cmap = cmap
        self._sleep = sleep
        self._axes = {}
        self._images_axes = {}
        self._1st_images = True
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

        # color bar setting
        self._cbar_ax = self._fig.add_axes([0.08, 0.83, 0.4, 0.02])
        norm = mpl.colors.Normalize(vmin=cbar_range[0], vmax=cbar_range[1])
        cb1 = mpl.colorbar.ColorbarBase(self._cbar_ax, cmap=self._cmap,
                                        norm=norm,
                                        orientation='horizontal',
                                        ticks=[self._cbar_range[0], np.mean(self._cbar_range), self._cbar_range[1]],
                                        label=r'$Weights\ value$')

        for r_loc, name in enumerate(objects):
            W_name = 'W_' + str(name+1)
            res_name = 'output_' + str(name+1)

            self._axes[W_name] = plt.subplot2grid(grid_space, (r_loc+1, 0),)
            self._axes[res_name] = plt.subplot2grid(grid_space, (r_loc+1, 1),)
            self._axes[W_name].tick_params(axis='both', which='both',
                                           bottom='off', top='off', left='off', right='off',
                                           labelbottom='off', labelleft='off',)
            self._axes[res_name].tick_params(axis='both', which='both',
                                             bottom='off', top='off', left='off', right='off',
                                             labelbottom='off', labelleft='off', )
            self._axes[W_name].grid(False)
            self._axes[res_name].grid(False)
            self._axes[W_name].set_title(r'$%s$' % W_name)
            self._axes[res_name].set_title(r'$%s$' % res_name)

            W_y_label = neural_structure.iloc[name+1]['input_size']
            W_x_label = neural_structure.iloc[name+1]['output_size']
            res_x_label = W_x_label
            self._axes[W_name].set_ylabel(r'$%i\ inputs$' % W_y_label)
            self._axes[W_name].set_xlabel(r'$%i\ outputs$' % W_x_label)
            self._axes[res_name].set_xlabel(r'$%i\ neurons$' % res_x_label)
        self._fig.subplots_adjust(hspace=0.4)

        plt.ion()
        plt.show()

    def monitoring(self, t_xs, t_ys):

        all_Ws = self._network.sess.run(list(self._network.Ws.values))
        feed_dict = self.evaluator.get_feed_dict(t_xs, t_ys)
        all_outputs = self._network.sess.run(list(self._network.layers_results['activated'].values)[1:],
                                             feed_dict=feed_dict)
        # for 1st layer
        res_name = 'input'
        if self._1st_images:
            self._images_axes[res_name] = self._axes[res_name].imshow(
                t_xs, interpolation='nearest', cmap=self._cmap, origin='lower')
        else:
            self._images_axes[res_name].set_data(t_xs)
        res_y_label = len(all_outputs[0])
        self._axes[res_name].set_ylabel(r'$batch\ size:%i$' % res_y_label)
        for name in self._objects:
            W_name = 'W_' + str(name + 1)
            res_name = 'output_' + str(name + 1)
            if self._1st_images:
                self._images_axes[W_name] = self._axes[W_name].imshow(all_Ws[name], interpolation='nearest',
                                                    vmin=self._cbar_range[0], vmax=self._cbar_range[1],
                                                    cmap=self._cmap, origin='lower')
                self._images_axes[res_name] = self._axes[res_name].imshow(all_outputs[name], interpolation='nearest',
                                                cmap=self._cmap, origin='lower',)
            else:
                self._images_axes[W_name].set_data(all_Ws[name])
                self._images_axes[res_name].set_data(all_outputs[name])
                # self._fig.canvas.blit(self._axes[W_name].bbox)
                # self._fig.canvas.blit(self._axes[res_name].bbox)
            res_y_label = len(all_outputs[name])
            self._axes[res_name].set_ylabel(r'$batch\ size:%i$' % res_y_label)
        # self._fig.canvas.draw()
        self._fig.canvas.flush_events()
        self._1st_images = False
        plt.pause(self._sleep)



