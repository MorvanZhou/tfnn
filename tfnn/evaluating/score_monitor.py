import matplotlib.pyplot as plt
import pandas as pd
from tfnn.evaluating.monitor import Monitor


class ScoreMonitor(Monitor):
    def __init__(self, grid_space, objects, evaluator):
        super(ScoreMonitor, self).__init__(evaluator, 'score_monitor')
        self._axes = {}
        fig = plt.figure()
        for r_loc, name in enumerate(objects):
            r_span, c_span = 1, grid_space[1]
            self._axes[name] = plt.subplot2grid(grid_space, (r_loc, 0), colspan=c_span, rowspan=r_span)
            if name != objects[-1]:
                plt.setp(self._axes[name].get_xticklabels(), visible=False)
            self._axes[name].set_ylabel(r'$%s$' % name)
        fig.subplots_adjust(hspace=0.1)
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
            [self._axes[_name].plot([], [], c='r', ls='-', label='train') for _name in object_names]
            if v_xs is not None:
                [self._axes[_name].plot([], [], c='b', ls='--', label='test') for _name in object_names]
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

        if len(self._t_logs) == 2:
            for _name in object_names:
                self._axes[_name].plot(self._epoch, self._t_logs[_name].values, 'r-', label='train', )
            self._t_logs = self._t_logs.iloc[-1:, :]
            if v_xs is not None:
                for _name in object_names:
                    self._axes[_name].plot(self._epoch, self._v_logs[_name].values, 'b--', label='test', )
                self._v_logs = self._v_logs.iloc[-1:, :]
            self._epoch = self._epoch[-1:]
            plt.draw()
            plt.pause(0.01)


    @staticmethod
    def hold_plot():
        plt.ioff()
        plt.show()

