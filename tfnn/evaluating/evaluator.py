import tfnn
import matplotlib.pyplot as plt
import numpy as np
from tfnn.evaluating.scalar_monitor import ScaleMonitor
from tfnn.evaluating.layer_monitor import LayerMonitor
from tfnn.evaluating.data_fitting_monitor import DataFittingMonitor
from tfnn.evaluating.line_fitting_monitor import LineFittingMonitor
plt.style.use('ggplot')


class Evaluator(object):
    def __init__(self, network, ):
        self.network = network
        if isinstance(self.network, tfnn.RegNetwork):
            self._set_r2()
        if isinstance(self.network, tfnn.ClfNetwork):
            self._set_confusion_metrics()
            self._set_accuracy()

    def compute_scores(self, scores, xs, ys):
        if isinstance(scores, str):
            scores = [scores]
        if not isinstance(scores, (list, tuple)):
            raise TypeError('Scores must be a string or a tuple or a list of strings')
        scores_ops = []
        for score in scores:
            score = score.lower()
            if score == 'r2':
                scores_ops.append(self.r2)
            elif score == 'cost':
                scores_ops.append(self.network.loss)
            elif score == 'f1':
                scores_ops.append(self.f1)
            elif score == 'recall':
                scores_ops.append(self.recall)
            elif score == 'precision':
                scores_ops.append(self.precision)
            elif score == 'accuracy':
                scores_ops.append(self.accuracy)
            else:
                raise ValueError('Do not have %s score' % score)
        feed_dict = self.get_feed_dict(xs, ys)
        return self.network.sess.run(scores_ops, feed_dict=feed_dict)

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

    def set_scale_monitor(self, objects, figsize=(10, 10), sleep=0.001):
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
        self.scale_monitor = ScaleMonitor(grid_space, objects, self, figsize, sleep)
        return self.scale_monitor

    def set_layer_monitor(self, objects, figsize=(13, 10), cbar_range=(-1, 1), cmap='rainbow',
                          sleep=0.001):
        if isinstance(objects, (tuple, list)):
            grid_space = (2, len(objects)+1)
        else:
            raise ValueError("""objects should be a a list or dictionary. A list of layer index like
                                [0, 1, 3].
                                Not a %s""" % type(objects))
        self.layer_monitor = LayerMonitor(grid_space, objects, self, figsize, cbar_range, cmap, sleep)
        return self.layer_monitor

    def set_data_fitting_monitor(self, figsize=(8, 7), sleep=0.001):
        """
        Suitable for analysing the preprocessing with only one output unit.
        :param v_xs: validated xs
        :param v_ys: validated ys (single attribute)
        :param continue_plot: True or False
        :return: Plotting
        """
        if not isinstance(self.network, tfnn.RegNetwork):
            raise NotImplementedError('Can only plot for Regression neural network.')
        self.data_fitting_monitor = DataFittingMonitor(self, figsize, sleep)
        return self.data_fitting_monitor

    def set_line_fitting_monitor(self, figsize=(8, 7), sleep=0.001):
        if not isinstance(self.network, tfnn.RegNetwork):
            raise NotImplementedError('Can only plot this result for Regression neural network.')
        self.line_fitting_monitor = LineFittingMonitor(self, figsize, sleep)
        return self.line_fitting_monitor

    def monitoring(self, t_xs, t_ys, **kwargs):
        if hasattr(self, 'scale_monitor'):
            v_xs, v_ys = kwargs['v_xs'], kwargs['v_ys']
            self.scale_monitor.monitoring(t_xs, t_ys, v_xs, v_ys)
        if hasattr(self, 'layer_monitor'):
            self.layer_monitor.monitoring(t_xs, t_ys)
        if hasattr(self, 'data_fitting_monitor'):
            v_xs, v_ys = kwargs['v_xs'], kwargs['v_ys']
            self.data_fitting_monitor.monitoring(v_xs, v_ys)
        if hasattr(self, 'line_fitting_monitor'):
            self.line_fitting_monitor.monitoring(t_xs, t_ys)

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
        print('Press any key to exit...')
        plt.ioff()
        plt.waitforbuttonpress()
        plt.close()

    def _set_accuracy(self):
        if isinstance(self.network, tfnn.ClfNetwork):
            with tfnn.name_scope('accuracy'):
                correct_prediction = tfnn.equal(
                    tfnn.argmax(self.network.predictions, 1),
                    tfnn.argmax(self.network.target_placeholder, 1),
                    name='correct_prediction')
                self.accuracy = tfnn.reduce_mean(
                    tfnn.cast(correct_prediction, tfnn.float32), name='accuracy')
                tfnn.scalar_summary('accuracy', self.accuracy)

    def _set_r2(self):
        if isinstance(self.network, tfnn.RegNetwork):
            with tfnn.name_scope('r2_score'):
                self.ys_mean = ys_mean = tfnn.reduce_mean(self.network.target_placeholder,
                                                          reduction_indices=[0],
                                                          name='ys_mean')
                self.ss_tot = ss_tot = tfnn.reduce_sum(
                    tfnn.square(self.network.target_placeholder - ys_mean),
                    reduction_indices=[0], name='total_sum_squares')
                # ss_reg = np.sum(np.square(predictions-ys_mean), axis=0)
                self.ss_res = ss_res = tfnn.reduce_sum(
                    tfnn.square(self.network.target_placeholder - self.network.predictions),
                    reduction_indices=[0], name='residual_sum_squares')
                self.aaa = ss_res / ss_tot
                self.r2 = tfnn.reduce_mean(
                    tfnn.sub(tfnn.ones_like(ss_res, dtype=tfnn.float32), (ss_res / ss_tot)),
                    name='coefficient_of_determination')
                tfnn.scalar_summary('r2_score', self.r2)

    def _set_confusion_metrics(self):
        # from https://cloud.google.com/solutions/machine-learning-with-financial-time-series-data
        # for onehot data
        with tfnn.name_scope('f1_score'):
            predictions = tfnn.argmax(self.network.predictions, 1)
            actuals = tfnn.argmax(self.network.target_placeholder, 1)

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

            self.recall = tp / (tp + fn)
            self.precision = tp / (tp + fp)

            self.f1 = tfnn.div(2 * (self.precision * self.recall),
                               (self.precision + self.recall), name='f1_score')
            tfnn.scalar_summary('f1_score', self.f1)
            tfnn.scalar_summary('precision', self.precision)
            tfnn.scalar_summary('recall', self.recall)