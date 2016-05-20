import tensorflow as tf
import tfnn


class AccuracyEvaluator(object):
    def __init__(self, network, ):
        self.network = network
        if isinstance(self.network, tfnn.ClassificationNetwork):
            with tf.name_scope('accuracy'):
                correct_prediction = tf.equal(tf.argmax(network.predictions, 1),
                                              tf.argmax(network.target_placeholder, 1), name='correct_prediction')
                self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
                tf.scalar_summary('accuracy', self.accuracy)

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

    def plot_comparison(self, v_xs, v_ys):
        if not isinstance(self.network, tfnn.RegressionNetwork):
            raise NotImplementedError('Can only compute accuracy for Classification neural network.')
