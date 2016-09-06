import tfnn
import matplotlib.pyplot as plt


class Plotter(object):
    def __init__(self, network):
        self._network = network
        if isinstance(self._network, tfnn.RegNetwork):
            self.first_time_lm = True
            self.first_time_soc = True

    def plot_regression_linear_comparison(self, y_real, y_predict, continue_plot,):
        if self.first_time_soc:
            y_real_max, y_real_min = y_real.min(), y_real.max()
            self.first_time_soc = False
            self.fig_soc, self.ax_soc = plt.subplots()
            self.scat_soc = self.ax_soc.scatter(y_real, y_predict, label='predicted', alpha=0.5)
            self.ax_soc.plot([y_real_min, y_real_max], [y_real_min, y_real_max], 'r--', lw=4, label='real')
            self.ax_soc.grid(True)
            self.ax_soc.legend(loc='upper left')
            self.ax_soc.set_xlabel('Real data')
            offset = 0.1 * (y_real_max[0] - y_real_min[0])
            self.ax_soc.set_ylim([y_real_min[0] - offset, y_real_max[0] + offset])
            self.ax_soc.set_xlim([y_real_min[0] - offset, y_real_max[0] + offset])
            self.ax_soc.set_ylabel('Predicted')
            if continue_plot:
                plt.ion()
            plt.show()
        else:
            plt.pause(0.02)
            self.scat_soc.remove()
            self.scat_soc = self.ax_soc.scatter(y_real, y_predict, label='predicted', alpha=0.5)
            plt.draw()

    def plot_regression_nonlinear_comparison(self, xs, ys, y_predict, continue_plot=False):
        if self.first_time_lm:
            self.first_time_lm = False
            self.fig_lm, self.ax_lm = plt.subplots()
            self.ax_lm.scatter(xs, ys, c='red', s=20, alpha=0.5)
            self.scat_lm = self.ax_lm.scatter(xs, y_predict, c='blue', s=20, alpha=0.5)
            self.ax_lm.set_xlabel('Input')
            self.ax_lm.set_ylabel('Output')
            if continue_plot:
                plt.ion()
            plt.show()
        else:
            plt.pause(0.02)
            self.scat_lm.remove()
            self.scat_lm = self.ax_lm.scatter(xs, y_predict, c='blue', s=20, alpha=0.5)
            plt.draw()

    def plot_instant_cost_r2(self, t_cost, t_r2, global_step, v_cost=None, v_r2=None):
        if not hasattr(self, '_t_cost_logs'):
            self._epoch = []
            self._t_cost_logs = []
            self._v_cost_logs = []
            self._t_r2_logs = []
            self._v_r2_logs = []
            if not hasattr(self, 'figure'):
                self.figure = plt.figure()
                self.axs = [self.figure.add_subplot(2, 1, i) for i in [1, 2]]
            self.axs[0].set_ylabel('R2 Score')
            self.axs[1].set_ylabel('Cost')

            self.axs[1].set_xlabel('Epoch')
            self.axs[0].get_xaxis().set_ticklabels([])
            [self.axs[i].plot([], [], c='r', ls='-', label='train') for i in [0, 1]]
            if v_cost is not None:
                [self.axs[i].plot([], [], c='b', ls='--', label='test') for i in [0, 1]]
            self.axs[0].legend(loc='lower right')
            self.axs[1].legend(loc='upper right')
            # self.axs[0].set_ylim(top=1, bottom=-5)
            # self.axs[1].set_ylim(bottom=0)
            plt.ion()
            plt.show()
            plt.pause(0.01)

        self._epoch.append(global_step)
        self._t_cost_logs.append(t_cost)
        self._t_r2_logs.append(t_r2)
        if (v_cost is not None) and (v_r2 is not None):
            self._v_cost_logs.append(v_cost)
            self._v_r2_logs.append(v_r2)

        if len(self._t_cost_logs) == 2:
            self.axs[0].plot(self._epoch, self._t_r2_logs, 'r-', label='train', )
            self.axs[1].plot(self._epoch, self._t_cost_logs, 'r-', label='train',)
            self._t_cost_logs = self._t_cost_logs[-1:]
            self._t_r2_logs = self._t_r2_logs[-1:]
            if v_cost is not None:
                self.axs[0].plot(self._epoch, self._v_r2_logs, 'b--', label='test', )
                self.axs[1].plot(self._epoch, self._v_cost_logs, 'b--', label='test',)
                self._v_cost_logs = self._v_cost_logs[-1:]
                self._v_r2_logs = self._v_r2_logs[-1:]
            self._epoch = self._epoch[-1:]
            plt.draw()
            plt.pause(0.02)