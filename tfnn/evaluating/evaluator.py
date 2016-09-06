import tfnn
import matplotlib.pyplot as plt
from tfnn.evaluating.plotter import Plotter
plt.style.use('ggplot')


class Evaluator(object):
    def __init__(self, network, ):
        self._network = network
        self._plotter = Plotter(self._network)
        if isinstance(self._network, tfnn.ClfNetwork):
            with tfnn.name_scope('accuracy'):
                correct_prediction = tfnn.equal(tfnn.argmax(network.predictions, 1),
                                                tfnn.argmax(network.target_placeholder, 1), name='correct_prediction')
                self._accuracy = tfnn.reduce_mean(tfnn.cast(correct_prediction, tfnn.float32), name='accuracy')
                tfnn.scalar_summary('accuracy', self._accuracy)

        with tfnn.name_scope('r2_score'):
            ys_mean = tfnn.reduce_mean(network.target_placeholder, reduction_indices=[0], name='ys_mean')
            ss_tot = tfnn.reduce_sum(tfnn.square(network.target_placeholder - ys_mean),
                                     reduction_indices=[0], name='total_sum_squares')
            # ss_reg = np.sum(np.square(predictions-ys_mean), axis=0)
            ss_res = tfnn.reduce_sum(tfnn.square(network.target_placeholder - network.predictions),
                                     reduction_indices=[0], name='residual_sum_squares')
            self._r2 = tfnn.sub(tfnn.constant(1, dtype=tfnn.float32), (ss_res / ss_tot)[0],
                                name='coefficient_of_determination')
            tfnn.scalar_summary('r2_score', self._r2)

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

            self._f1 = tfnn.div(2 * (precision * recall), (precision + recall), name='f1_score')
            tfnn.scalar_summary('f1_score', self._f1)

    def compute_r2(self, xs, ys):
        feed_dict = self._get_feed_dict(xs, ys)
        return self._r2.eval(feed_dict, self._network.sess)

    def compute_accuracy(self, xs, ys):
        # ignore dropout and regularization
        if not isinstance(self._network, tfnn.ClfNetwork):
            raise NotImplementedError('Can only compute accuracy for Classification neural network.')
        feed_dict = self._get_feed_dict(xs, ys)
        return self._accuracy.eval(feed_dict, self._network.sess)

    def compute_cost(self, xs, ys):
        feed_dict = self._get_feed_dict(xs, ys)
        return self._network.loss.eval(feed_dict, self._network.sess)

    def compute_f1(self, xs, ys):
        feed_dict = self._get_feed_dict(xs, ys)
        return self._f1.eval(feed_dict, self._network.sess)

    def monitoring(self, xs, ys,
                   results=[['r2'],
                            ['cost']],
                   v_xs=None, v_ys=None):
        pass

    def plot_regression_linear_comparison(self, xs, ys, continue_plot=False):
        """
        Suitable for analysing the datasets with only one output unit.
        :param v_xs: validated xs
        :param v_ys: validated ys (single attribute)
        :param continue_plot: True or False
        :return: Plotting
        """
        if not isinstance(self._network, tfnn.RegNetwork):
            raise NotImplementedError('Can only plot for Regression neural network.')
        elif ys.shape[1] > 1:
            raise NotImplementedError('Can only support ys which have single value.')
        feed_dict = self._get_feed_dict(xs, ys)
        predictions = self._network.predictions.eval(feed_dict, self._network.sess)
        self._plotter.plot_regression_linear_comparison(ys, predictions, continue_plot)

    def plot_regression_nonlinear_comparison(self, xs, ys, continue_plot=False):
        """
        Suitable for analysing the dataset with only one attribute and single output.
        :param v_xs: Only has one attribute
        :param v_ys: Only has one attribute
        :param continue_plot: True or False
        :return: plotting
        """
        if not isinstance(self._network, tfnn.RegNetwork):
            raise NotImplementedError('Can only plot this result for Regression neural network.')
        elif ys.shape[1] > 1:
            raise NotImplementedError('Can only support ys which have single value.')
        feed_dict = self._get_feed_dict(xs, ys)
        predictions = self._network.predictions.eval(feed_dict, self._network.sess)
        self._plotter.plot_regression_nonlinear_comparison(xs, ys, predictions, continue_plot)

    def plot_instant_cost_r2(self, t_xs, t_ys, global_step, v_xs=None, v_ys=None):
        t_cost = self.compute_cost(t_xs, t_ys)
        t_r2 = self.compute_r2(t_xs, t_ys)
        if (v_xs is not None) and (v_ys is not None):
            v_cost = self.compute_cost(v_xs, v_ys)
            v_r2 = self.compute_r2(v_xs, v_ys)
        elif (v_xs is None) and (v_ys is None):
            v_cost, v_r2 = [None] * 2
        else:
            raise ValueError('If plot test data, the v_xs, v_ys cannot be None')
        self._plotter.plot_instant_cost_r2(t_cost, t_r2, global_step, v_cost, v_r2)

    @staticmethod
    def hold_plot():
        plt.ioff()
        plt.show()

    def _get_feed_dict(self, xs, ys):
        if self._network.reg == 'dropout':
            feed_dict = {self._network.data_placeholder: xs,
                         self._network.target_placeholder: ys,
                         self._network.keep_prob_placeholder: 1.}
        elif self._network.reg == 'l2':
            feed_dict = {self._network.data_placeholder: xs,
                         self._network.target_placeholder: ys,
                         self._network.l2_placeholder: 0.}
        else:
            feed_dict = {self._network.data_placeholder: xs,
                         self._network.target_placeholder: ys}
        return feed_dict

if __name__ == '__main__':
    a = tfnn.size(tfnn.ones((2,2)))
    with tfnn.Session() as sess:
        print(sess.run(a))