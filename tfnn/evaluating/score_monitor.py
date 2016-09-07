import matplotlib.pyplot as plt
import pandas as pd
from tfnn.evaluating.monitor import Monitor


class ScoreMonitor(Monitor):
    def __init__(self, grid_space, objects, evaluator, figsize, sleep=0.001):
        super(ScoreMonitor, self).__init__(evaluator, 'score_monitor')
        self._axes = {}
        self._tplot_axes = {}
        self._vplot_axes = {}
        self._fig = plt.figure(figsize=figsize)
        self._sleep = sleep
        self._1st_plot = True
        for r_loc, name in enumerate(objects):
            r_span, c_span = 1, grid_space[1]
            self._axes[name] = plt.subplot2grid(grid_space, (r_loc, 0), colspan=c_span, rowspan=r_span)
            if name != objects[-1]:
                plt.setp(self._axes[name].get_xticklabels(), visible=False)
            self._axes[name].set_ylabel(r'$%s$' % name)
        self._fig.subplots_adjust(hspace=0.1)
        plt.ion()
        plt.show()

    def monitoring(self, t_xs, t_ys, global_step, v_xs=None, v_ys=None):
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
            else:
                raise ValueError('No object name as %s' % object_name)
        t_feed_dict = self.evaluator.get_feed_dict(t_xs, t_ys)
        results = self.evaluator.network.sess.run(object_ops, t_feed_dict)
        t_results = {}
        for name, res in zip(object_names, results):
            t_results[name] = res

        if (v_xs is not None) and (v_ys is not None):
            v_feed_dict = self.evaluator.get_feed_dict(v_xs, v_ys)
            results = self.evaluator.network.sess.run(object_ops, v_feed_dict)
            v_results = {}
            for name, res in zip(object_names, results):
                v_results[name] = res

        if not hasattr(self, '_t_logs'):
            self._epoch = [global_step]
            self._t_logs = pd.DataFrame(columns=object_names)
            self._t_logs = self._t_logs.append(t_results, ignore_index=True)
            if v_xs is not None:
                self._v_logs = pd.DataFrame(columns=object_names)
                self._v_logs = self._v_logs.append(v_results, ignore_index=True)
            for _name in object_names:
                self._tplot_axes[_name], = self._axes[_name].plot([1, 1], [1, 1],
                                                                  c='#9999ff',   # red like
                                                                  ls='-',
                                                                  lw=2, label='train')
            if v_xs is not None:
                for _name in object_names:
                    self._vplot_axes[_name], = self._axes[_name].plot([2, 2], [2, 2],
                                                                      c='#ff9999',  # blue like
                                                                      ls='--',
                                                                      lw=2, label='test')
            for _name in object_names:
                if _name in ['r2', 'accuracy', 'f1']:
                    self._axes[_name].legend(loc='lower right')
                elif _name == 'cost':
                    self._axes[_name].legend(loc='upper right')
            plt.pause(0.01)

        self._epoch.append(global_step)
        self._t_logs = self._t_logs.append(t_results, ignore_index=True)

        if v_xs is not None:
            self._v_logs = self._v_logs.append(v_results, ignore_index=True)

        if len(self._t_logs) >= 2:
            for _name in object_names:
                self._tplot_axes[_name].set_xdata(self._epoch)
                self._tplot_axes[_name].set_ydata(self._t_logs[_name].values)
                if v_xs is not None:
                    self._vplot_axes[_name].set_data(self._epoch, self._v_logs[_name].values)
                self._axes[_name].relim()
                self._axes[_name].autoscale_view()
            self._fig.canvas.draw()
            self._fig.canvas.flush_events()
            plt.pause(self._sleep)


