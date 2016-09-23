import matplotlib.pyplot as plt
from tfnn.evaluating.monitor import Monitor
import numpy as np


class ScaleMonitor(Monitor):
    def __init__(self, grid_space, objects, evaluator, figsize, sleep=0.001):
        super(ScaleMonitor, self).__init__(evaluator, 'score_monitor')
        self._network = self.evaluator.network
        self._axes = {}
        self._tplot_axes = {}
        self._vplot_axes = {}
        self._fig = plt.figure(figsize=figsize)
        self._sleep = sleep
        for r_loc, name in enumerate(objects):
            r_span, c_span = 1, grid_space[1]
            self._axes[name] = plt.subplot2grid(grid_space, (r_loc, 0), colspan=c_span, rowspan=r_span)
            if name != objects[-1]:
                plt.setp(self._axes[name].get_xticklabels(), visible=False)
            self._axes[name].set_ylabel(r'$%s$' % name.replace(' ', r'\ ').capitalize())
        self._fig.subplots_adjust(hspace=0.1)
        plt.ion()
        plt.show()

    def monitoring(self, t_xs, t_ys, v_xs=None, v_ys=None):
        object_ops, object_names = self._get_object_ops()
        t_results, v_results = self._get_results(t_xs, t_ys, v_xs, v_ys, object_ops)
        global_step = self._network.sess.run([self._network.global_step])
        if not hasattr(self, '_t_logs'):
            self._plot_1st_frame(global_step, t_results, v_results, object_names)
        else:
            self._plot_rest_frames(global_step, t_results, v_results, object_names)
            plt.pause(self._sleep)

    def _get_object_ops(self):
        object_ops = []
        object_names = []
        for object_name in self._axes.keys():
            if object_name == 'r2':
                object_names.append(object_name)
                object_ops.append(self.evaluator.r2)
            elif object_name == 'accuracy':
                object_names.append(object_name)
                object_ops.append(self.evaluator.accuracy)
            elif object_name == 'cost':
                object_names.append(object_name)
                object_ops.append(self.evaluator.network.loss)
            elif object_name == 'f1':
                object_names.append(object_name)
                object_ops.append(self.evaluator.f1)
            elif object_name == 'precision':
                object_names.append(object_name)
                object_ops.append(self.evaluator.precision)
            elif object_name == 'recall':
                object_names.append(object_name)
                object_ops.append(self.evaluator.recall)
            elif object_name == 'learning rate':
                object_names.append(object_name)
                object_ops.append(self._network.lr)
            elif object_name == 'dropout':
                if not hasattr(self._network, 'keep_prob'):
                    raise AttributeError('The dropout has not been set.')
                object_names.append(object_name)
                object_ops.append(self._network.keep_prob)
            else:
                raise ValueError('No object name as %s' % object_name)
        return [object_ops, object_names]

    def _get_results(self, t_xs, t_ys, v_xs, v_ys, object_ops):
        t_feed_dict = self.evaluator.get_feed_dict(t_xs, t_ys)
        # t_results has the order of object_names
        t_results = self._network.sess.run(object_ops, t_feed_dict)

        if (v_xs is not None) and (v_ys is not None):
            v_feed_dict = self.evaluator.get_feed_dict(v_xs, v_ys)
            # v_results has the order of object_names
            v_results = self._network.sess.run(object_ops, v_feed_dict)
        else:
            v_results = None
        return [t_results, v_results]

    def _plot_1st_frame(self, global_step, t_results, v_results, object_names):
        self._epoch = [global_step]
        self._t_logs = np.array([t_results])
        if v_results is not None:
            self._v_logs = np.array([v_results])
        for _name in object_names:
            self._tplot_axes[_name], = self._axes[_name].plot([1, 1], [1, 1],
                                                              c=self.color_train,  # red like
                                                              ls='-',
                                                              lw=2, label=r'$Train$')
            if v_results is not None:
                if _name not in ['learning rate', 'dropout']:
                    self._vplot_axes[_name], = self._axes[_name].plot([2, 2], [2, 2],
                                                                      c=self.color_test,  # blue like
                                                                      ls='--',
                                                                      lw=2, label=r'$Test$')
        for _name in object_names:
            if _name in ['r2', 'accuracy', 'f1', 'recall', 'precision']:
                self._axes[_name].legend(loc='lower right')
            elif _name == 'cost':
                self._axes[_name].legend(loc='upper right')

    def _plot_rest_frames(self, global_step, t_results, v_results, object_names):
        self._epoch.append(global_step)
        self._t_logs = np.vstack((self._t_logs, t_results))
        if v_results is not None:
            self._v_logs = np.vstack((self._v_logs, v_results))

        if len(self._t_logs) >= 2:
            for _index, _name in enumerate(object_names):
                self._tplot_axes[_name].set_data(self._epoch, self._t_logs[:, _index])
                if v_results is not None:
                    if _name not in ['learning rate', 'dropout']:
                        self._vplot_axes[_name].set_data(self._epoch, self._v_logs[:, _index])
                self._axes[_name].relim()
                self._axes[_name].autoscale_view()
            self._fig.canvas.draw()
            self._fig.canvas.flush_events()
