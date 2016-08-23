import tfnn
import matplotlib.pyplot as plt
plt.style.use('ggplot')


class Evaluator(object):
    def __init__(self, network, ):
        self.network = network
        if isinstance(self.network, tfnn.ClfNetwork):
            with tfnn.name_scope('accuracy'):
                with tfnn.name_scope('correct_prediction'):
                    correct_prediction = tfnn.equal(tfnn.argmax(network.predictions, 1),
                                                  tfnn.argmax(network.target_placeholder, 1), name='correct_prediction')
                with tfnn.name_scope('accuracy'):
                    self.accuracy = tfnn.reduce_mean(tfnn.cast(correct_prediction, tfnn.float32), name='accuracy')
                tfnn.scalar_summary('accuracy', self.accuracy)
        elif isinstance(self.network, tfnn.RegNetwork):
            self.first_time_lm = True
            self.first_time_soc = True
        with tfnn.name_scope('r2_score'):
            with tfnn.name_scope('ys_mean'):
                ys_mean = tfnn.reduce_mean(network.target_placeholder, reduction_indices=[0], name='ys_mean')
            with tfnn.name_scope('total_sum_squares'):
                ss_tot = tfnn.reduce_sum(tfnn.square(network.target_placeholder - ys_mean),
                                         reduction_indices=[0], name='total_sum_squares')
            # ss_reg = np.sum(np.square(predictions-ys_mean), axis=0)
            with tfnn.name_scope('residual_sum_squares'):
                ss_res = tfnn.reduce_sum(tfnn.square(network.target_placeholder - network.predictions),
                                         reduction_indices=[0], name='residual_sum_squares')
            with tfnn.name_scope('coefficient_of_determination'):
                self.r2_score = tfnn.sub(tfnn.constant(1, dtype=tfnn.float32), (ss_res / ss_tot)[0],
                                         name='coefficient_of_determination')
            tfnn.scalar_summary('r2_score', self.r2_score)

    def compute_r2_score(self, xs, ys):
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
        return self.r2_score.eval(feed_dict, self.network.sess)

    def compute_accuracy(self, xs, ys):
        if not isinstance(self.network, tfnn.ClfNetwork):
            raise NotImplementedError('Can only compute accuracy for Classification neural network.')

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

        return self.accuracy.eval(feed_dict, self.network.sess)

    def compute_cost(self, xs, ys):
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
        return self.network.loss.eval(feed_dict, self.network.sess)

    def regression_plot_linear_comparison(self, v_xs, v_ys, continue_plot=False):
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
        if self.network.reg == 'dropout':
            feed_dict = {self.network.data_placeholder: v_xs,
                         self.network.target_placeholder: v_ys,
                         self.network.keep_prob_placeholder: 1.}
        elif self.network.reg == 'l2':
            feed_dict = {self.network.data_placeholder: v_xs,
                         self.network.target_placeholder: v_ys,
                         self.network.l2_placeholder: 0.}
        else:
            feed_dict = {self.network.data_placeholder: v_xs,
                         self.network.target_placeholder: v_ys}
        predictions = self.network.predictions.eval(feed_dict, self.network.sess)

        if self.first_time_soc:
            self.first_time_soc = False
            self.fig_soc, self.ax_soc = plt.subplots()
            self.scat_soc = self.ax_soc.scatter(v_ys, predictions, label='predicted')
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
            self.scat_soc = self.ax_soc.scatter(v_ys, predictions, label='predicted')
            plt.draw()

    def regression_plot_nonlinear_comparison(self, v_xs, v_ys, continue_plot=False):
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
        if self.network.reg == 'dropout':
            feed_dict = {self.network.data_placeholder: v_xs,
                         self.network.target_placeholder: v_ys,
                         self.network.keep_prob_placeholder: 1.}
        elif self.network.reg == 'l2':
            feed_dict = {self.network.data_placeholder: v_xs,
                         self.network.target_placeholder: v_ys,
                         self.network.l2_placeholder: 0.}
        else:
            feed_dict = {self.network.data_placeholder: v_xs,
                         self.network.target_placeholder: v_ys}
        predictions = self.network.predictions.eval(feed_dict, self.network.sess)
        if self.first_time_lm:
            self.first_time_lm = False
            self.fig_lm, self.ax_lm = plt.subplots()
            self.ax_lm.scatter(v_xs, v_ys, c='red', s=20)
            self.scat_lm = self.ax_lm.scatter(v_xs, predictions, c='blue', s=20)
            self.ax_lm.set_xlabel('Input')
            self.ax_lm.set_ylabel('Output')
            if continue_plot:
                plt.ion()
            plt.show()
        else:
            plt.pause(0.1)
            self.scat_lm.remove()
            self.scat_lm = self.ax_lm.scatter(v_xs, predictions, c='blue', s=20)

            plt.draw()

