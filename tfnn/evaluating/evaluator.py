import tfnn
import matplotlib.pyplot as plt
plt.style.use('ggplot')


class Evaluator(object):
    def __init__(self, network, ):
        self.network = network
        if isinstance(self.network, tfnn.ClfNetwork):
            with tfnn.name_scope('accuracy'):
                correct_prediction = tfnn.equal(tfnn.argmax(network.predictions, 1),
                                                tfnn.argmax(network.target_placeholder, 1), name='correct_prediction')
                self._accuracy = tfnn.reduce_mean(tfnn.cast(correct_prediction, tfnn.float32), name='accuracy')
                tfnn.scalar_summary('accuracy', self._accuracy)
        elif isinstance(self.network, tfnn.RegNetwork):
            self.first_time_lm = True
            self.first_time_soc = True
        with tfnn.name_scope('r2_score'):
            ys_mean = tfnn.reduce_mean(network.target_placeholder, reduction_indices=[0], name='ys_mean')
            ss_tot = tfnn.reduce_sum(tfnn.square(network.target_placeholder - ys_mean),
                                     reduction_indices=[0], name='total_sum_squares')
            # ss_reg = np.sum(np.square(predictions-ys_mean), axis=0)
            ss_res = tfnn.reduce_sum(tfnn.square(network.target_placeholder - network.predictions),
                                     reduction_indices=[0], name='residual_sum_squares')
            self._r2_score = tfnn.sub(tfnn.constant(1, dtype=tfnn.float32), (ss_res / ss_tot)[0],
                                     name='coefficient_of_determination')
            tfnn.scalar_summary('r2_score', self._r2_score)

    def compute_r2_score(self, xs, ys):
        feed_dict = self._get_feed_dict(xs, ys)
        return self._r2_score.eval(feed_dict, self.network.sess)

    def compute_accuracy(self, xs, ys):
        # ignore dropout and regularization
        if not isinstance(self.network, tfnn.ClfNetwork):
            raise NotImplementedError('Can only compute accuracy for Classification neural network.')
        feed_dict = self._get_feed_dict(xs, ys)
        return self._accuracy.eval(feed_dict, self.network.sess)

    def compute_cost(self, xs, ys):
        feed_dict = self._get_feed_dict(xs, ys)
        return self.network.loss.eval(feed_dict, self.network.sess)

    def plot_regression_linear_comparison(self, v_xs, v_ys, continue_plot=False):
        """
        Suitable for analysing the datasets with only one output unit.
        :param v_xs: validated xs
        :param v_ys: validated ys (single attribute)
        :param continue_plot: True or False
        :return: Plotting
        """
        if not isinstance(self.network, tfnn.RegNetwork):
            raise NotImplementedError('Can only plot for Regression neural network.')
        elif v_ys.shape[1] > 1:
            raise NotImplementedError('Can only support ys which have single value.')
        feed_dict = self._get_feed_dict(v_xs, v_ys)
        predictions = self.network.predictions.eval(feed_dict, self.network.sess)

        if self.first_time_soc:
            self.first_time_soc = False
            self.fig_soc, self.ax_soc = plt.subplots()
            self.scat_soc = self.ax_soc.scatter(v_ys, predictions, label='predicted', alpha=0.5)
            self.ax_soc.plot([v_ys.min(), v_ys.max()], [v_ys.min(), v_ys.max()], 'r--', lw=4, label='real')
            self.ax_soc.grid(True)
            self.ax_soc.legend(loc='upper left')
            self.ax_soc.set_xlabel('Real data')
            offset = 0.1 * (v_ys.max()[0] - v_ys.min()[0])
            self.ax_soc.set_ylim([v_ys.min()[0] - offset, v_ys.max()[0] + offset])
            self.ax_soc.set_xlim([v_ys.min()[0] - offset, v_ys.max()[0] + offset])
            self.ax_soc.set_ylabel('Predicted')
            if continue_plot:
                plt.ion()
            plt.show()
        else:
            plt.pause(0.1)
            self.scat_soc.remove()
            self.scat_soc = self.ax_soc.scatter(v_ys, predictions, label='predicted', alpha=0.5)
            plt.draw()

    def plot_regression_nonlinear_comparison(self, v_xs, v_ys, continue_plot=False):
        """
        Suitable for analysing the dataset with only one attribute and single output.
        :param v_xs: Only has one attribute
        :param v_ys: Only has one attribute
        :param continue_plot: True or False
        :return: plotting
        """
        if not isinstance(self.network, tfnn.RegNetwork):
            raise NotImplementedError('Can only plot this result for Regression neural network.')
        elif v_ys.shape[1] > 1:
            raise NotImplementedError('Can only support ys which have single value.')
        feed_dict = self._get_feed_dict(v_xs, v_ys)
        predictions = self.network.predictions.eval(feed_dict, self.network.sess)
        if self.first_time_lm:
            self.first_time_lm = False
            self.fig_lm, self.ax_lm = plt.subplots()
            self.ax_lm.scatter(v_xs, v_ys, c='red', s=20, alpha=0.5)
            self.scat_lm = self.ax_lm.scatter(v_xs, predictions, c='blue', s=20, alpha=0.5)
            self.ax_lm.set_xlabel('Input')
            self.ax_lm.set_ylabel('Output')
            if continue_plot:
                plt.ion()
            plt.show()
        else:
            plt.pause(0.1)
            self.scat_lm.remove()
            self.scat_lm = self.ax_lm.scatter(v_xs, predictions, c='blue', s=20, alpha=0.5)
            plt.draw()

    def plot_instant_cost_r2(self, t_xs, t_ys, global_step, v_xs=None, v_ys=None):
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
            [self.axs[i].plot([], [], c='b', ls='--', label='test') for i in [0, 1]]
            self.axs[0].legend(loc='lower right')
            self.axs[1].legend(loc='upper right')
            # self.axs[0].set_ylim(top=1, bottom=-5)
            # self.axs[1].set_ylim(bottom=0)
            plt.ion()
            plt.show()
            plt.pause(0.01)

        self._epoch.append(global_step)
        t_cost = self.compute_cost(t_xs, t_ys)
        t_r2 = self.compute_r2_score(t_xs, t_ys)
        self._t_cost_logs.append(t_cost)
        self._t_r2_logs.append(t_r2)
        if (v_xs is not None) and (v_ys is not None):
            v_cost = self.compute_cost(v_xs, v_ys)
            v_r2 = self.compute_r2_score(v_xs, v_ys)
            self._v_cost_logs.append(v_cost)
            self._v_r2_logs.append(v_r2)
        elif (v_xs is None) and (v_ys is None):
            pass
        else:
            raise ValueError('If plot test data, the v_xs, v_ys cannot be None')

        if len(self._t_cost_logs) == 2:
            self.axs[0].plot(self._epoch, self._t_r2_logs, 'r-', label='train', )
            self.axs[1].plot(self._epoch, self._t_cost_logs, 'r-', label='train',)
            self._t_cost_logs = self._t_cost_logs[-1:]
            self._t_r2_logs = self._t_r2_logs[-1:]
            if v_xs is not None:
                self.axs[0].plot(self._epoch, self._v_r2_logs, 'b--', label='test', )
                self.axs[1].plot(self._epoch, self._v_cost_logs, 'b--', label='test',)
                self._v_cost_logs = self._v_cost_logs[-1:]
                self._v_r2_logs = self._v_r2_logs[-1:]
            self._epoch = self._epoch[-1:]
            plt.draw()
            plt.pause(0.02)

    @staticmethod
    def hold_plot():
        plt.ioff()
        plt.show()

    def _get_feed_dict(self, xs, ys):
        if self.network.reg == 'dropout':
            feed_dict = {self.network.data_placeholder: xs,
                         self.network.target_placeholder: ys,
                         self.network.keep_prob_placeholder: 1.}
        elif self.network.reg == 'l2':
            feed_dict = {self.network.data_placeholder: xs,
                         self.network.target_placeholder: ys,
                         self.network.l2_placeholder: 0.}
        else:
            feed_dict = {self.network.data_placeholder: xs,
                         self.network.target_placeholder: ys}
        return feed_dict
