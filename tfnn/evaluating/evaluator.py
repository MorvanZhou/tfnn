import tfnn
import matplotlib.pyplot as plt


class Evaluator(object):
    def __init__(self, network, ):
        self.network = network
        if isinstance(self.network, tfnn.ClassificationNetwork):
            with tfnn.name_scope('accuracy'):
                correct_prediction = tfnn.equal(tfnn.argmax(network.predictions, 1),
                                              tfnn.argmax(network.target_placeholder, 1), name='correct_prediction')
                self.accuracy = tfnn.reduce_mean(tfnn.cast(correct_prediction, tfnn.float32), name='accuracy')
                tfnn.scalar_summary('accuracy', self.accuracy)
        elif isinstance(self.network, tfnn.RegressionNetwork):
            self.first_time = True

    def compute_accuracy(self, xs, ys):
        if not isinstance(self.network, tfnn.ClassificationNetwork):
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

    def plot_single_output_comparison(self, v_xs, v_ys, continue_plot=False):
        if not isinstance(self.network, tfnn.RegressionNetwork):
            raise NotImplementedError('Can only compute accuracy for Regression neural network.')
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
        fig, ax = plt.subplots()
        ax.scatter(v_ys, predictions)
        ax.plot([v_ys.min(), v_ys.max()], [v_ys.min(), v_ys.max()], 'k--', lw=2)
        ax.grid(True)
        ax.set_xlabel('Real data')
        ax.set_ylabel('Predicted')

        if self.first_time:
            self.first_time = False
            if continue_plot:
                plt.ion()
            plt.show()
        else:
            plt.pause(0.001)
            plt.close(fig)
            plt.draw()
