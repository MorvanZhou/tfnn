import tfnn
import matplotlib.pyplot as plt
from tfnn.evaluating.score_monitor import ScoreMonitor
from tfnn.evaluating.layer_monitor import LayerMonitor
plt.style.use('ggplot')


class Evaluator(object):
    def __init__(self, network, ):
        self.network = network
        if isinstance(self.network, tfnn.RegNetwork):
            self.first_time_lm = True
            self.first_time_soc = True

        if isinstance(self.network, tfnn.ClfNetwork):
            with tfnn.name_scope('accuracy'):
                correct_prediction = tfnn.equal(tfnn.argmax(network.predictions, 1),
                                                tfnn.argmax(network.target_placeholder, 1), name='correct_prediction')
                self.accuracy = tfnn.reduce_mean(tfnn.cast(correct_prediction, tfnn.float32), name='accuracy')
                tfnn.scalar_summary('accuracy', self.accuracy)
        if isinstance(self.network, tfnn.RegNetwork):
            with tfnn.name_scope('r2_score'):
                self.ys_mean = ys_mean = tfnn.reduce_mean(network.target_placeholder, reduction_indices=[0], name='ys_mean')
                self.ss_tot = ss_tot = tfnn.reduce_sum(tfnn.square(network.target_placeholder - ys_mean),
                                         reduction_indices=[0], name='total_sum_squares')
                # ss_reg = np.sum(np.square(predictions-ys_mean), axis=0)
                self.ss_res = ss_res = tfnn.reduce_sum(tfnn.square(network.target_placeholder - network.predictions),
                                         reduction_indices=[0], name='residual_sum_squares')
                self.aaa = ss_res/ss_tot
                self.r2 = tfnn.reduce_mean(
                    tfnn.sub(tfnn.ones_like(ss_res, dtype=tfnn.float32), (ss_res / ss_tot)),
                    name='coefficient_of_determination')
                tfnn.scalar_summary('r2_score', self.r2)

        if isinstance(self.network, tfnn.ClfNetwork):
            with tfnn.name_scope('f1_score'):
                predictions = tfnn.argmax(network.predictions, 1)
                actuals = tfnn.argmax(network.target_placeholder, 1)

                ones_like_actuals = tfnn.ones_like(actuals)
                zeros_like_actuals = tfnn.zeros_like(actuals)
                ones_like_predictions = tfnn.ones_like(predictions)
                zeros_like_predictions = tfnn.zeros_like(predictions)

                tp = tfnn.reduce_sum(
                    tfnn.cast(
                        tfnn.logical_and(
                            tfnn.equal(actuals, ones_like_actuals),
                            tfnn.equal(predictions, ones_like_predictions)
                        ), "float"))

                tn = tfnn.reduce_sum(
                    tfnn.cast(
                        tfnn.logical_and(
                            tfnn.equal(actuals, zeros_like_actuals),
                            tfnn.equal(predictions, zeros_like_predictions)
                        ), "float"))

                fp = tfnn.reduce_sum(
                    tfnn.cast(
                        tfnn.logical_and(
                            tfnn.equal(actuals, zeros_like_actuals),
                            tfnn.equal(predictions, ones_like_predictions)
                        ), "float"))

                fn = tfnn.reduce_sum(
                    tfnn.cast(
                        tfnn.logical_and(
                            tfnn.equal(actuals, ones_like_actuals),
                            tfnn.equal(predictions, zeros_like_predictions)
                        ), "float"))

                recall = tp / (tp + fn)
                precision = tp / (tp + fp)

                self.f1 = tfnn.div(2 * (precision * recall), (precision + recall), name='f1_score')
                tfnn.scalar_summary('f1_score', self.f1)

    def compute_r2(self, xs, ys):
        feed_dict = self.get_feed_dict(xs, ys)
        return self.r2.eval(feed_dict, self.network.sess)

    def compute_accuracy(self, xs, ys):
        # ignore dropout and regularization
        if not isinstance(self.network, tfnn.ClfNetwork):
            raise NotImplementedError('Can only compute accuracy for Classification neural network.')
        feed_dict = self.get_feed_dict(xs, ys)
        return self.accuracy.eval(feed_dict, self.network.sess)

    def compute_cost(self, xs, ys):
        feed_dict = self.get_feed_dict(xs, ys)
        return self.network.loss.eval(feed_dict, self.network.sess)

    def compute_f1(self, xs, ys):
        feed_dict = self.get_feed_dict(xs, ys)
        return self.f1.eval(feed_dict, self.network.sess)

    def set_score_monitor(self, objects, figsize=(10, 10), sleep=0.001):
        """
        :param objects: a list. A list like ['cost', 'r2'];
        :param grid_space: a tuple or list of (max_rows, max_cols);
        :return: Monitor
        """
        if isinstance(objects, (tuple, list)):
            grid_space = (len(objects), 1)
        else:
            raise ValueError("""objects should be a a list or dictionary. A list like ['cost', 'r2'].
                                Not a %s""" % type(objects))
        if isinstance(self.network, tfnn.ClfNetwork):
            if 'r2' in objects:
                raise ValueError('r2 score is not used for classification networks')
        if isinstance(self.network, tfnn.RegNetwork):
            if ('accuracy' in objects) or ('f1' in objects):
                raise ValueError('accuracy or f1 score are not used for regression networks')
        self.score_monitor = ScoreMonitor(grid_space, objects, self, figsize, sleep)
        return self.score_monitor

    def set_layer_monitor(self, objects, figsize=(13, 13), cbar_range=(-1, 1), cmap='rainbow',
                          sleep=0.001):
        if isinstance(objects, (tuple, list)):
            grid_space = (len(objects)+1, 2)
        else:
            raise ValueError("""objects should be a a list or dictionary. A list of layer index like
                                [0, 1, 3].
                                Not a %s""" % type(objects))
        self.layer_monitor = LayerMonitor(grid_space, objects, self, figsize, cbar_range, cmap, sleep)
        return self.layer_monitor

    def monitoring(self, t_xs, t_ys, global_step=None, v_xs=None, v_ys=None):
        if hasattr(self, 'score_monitor'):
            if global_step is None:
                raise ValueError('Pass global_step to this monitoring function')
            self.score_monitor.monitoring(t_xs, t_ys, global_step, v_xs, v_ys)
        if hasattr(self, 'layer_monitor'):
            self.layer_monitor.monitoring(t_xs, t_ys)

    def plot_regression_linear_comparison(self, xs, ys, continue_plot=False):
        """
        Suitable for analysing the datasets with only one output unit.
        :param v_xs: validated xs
        :param v_ys: validated ys (single attribute)
        :param continue_plot: True or False
        :return: Plotting
        """
        if not isinstance(self.network, tfnn.RegNetwork):
            raise NotImplementedError('Can only plot for Regression neural network.')
        elif ys.shape[1] > 1:
            raise NotImplementedError('Can only support ys which have single value.')
        feed_dict = self.get_feed_dict(xs, ys)
        y_predict = self.network.predictions.eval(feed_dict, self.network.sess)
        if self.first_time_soc:
            y_real_max, y_real_min = ys.min(), ys.max()
            self.first_time_soc = False
            self.fig_soc, self.ax_soc = plt.subplots()
            self.scat_soc = self.ax_soc.scatter(ys, y_predict, label='predicted', alpha=0.5)
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
            self.scat_soc = self.ax_soc.scatter(ys, y_predict, label='predicted', alpha=0.5)
            plt.draw()

    def plot_regression_nonlinear_comparison(self, xs, ys, continue_plot=False):
        """
        Suitable for analysing the dataset with only one attribute and single output.
        :param v_xs: Only has one attribute
        :param v_ys: Only has one attribute
        :param continue_plot: True or False
        :return: plotting
        """
        if not isinstance(self.network, tfnn.RegNetwork):
            raise NotImplementedError('Can only plot this result for Regression neural network.')
        elif ys.shape[1] > 1:
            raise NotImplementedError('Can only support ys which have single value.')
        feed_dict = self.get_feed_dict(xs, ys)
        y_predict = self.network.predictions.eval(feed_dict, self.network.sess)
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

    def get_feed_dict(self, xs, ys):
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

    @staticmethod
    def hold_plot():
        plt.ioff()
        plt.show()