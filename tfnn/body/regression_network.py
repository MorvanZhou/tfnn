from tfnn.body.network import Network
import tensorflow as tf


class RegressionNetwork(Network):
    def __init__(self, n_inputs, n_outputs, intput_dtype=tf.float32, output_dtype=tf.float32, l2=0, seed=None):
        super(RegressionNetwork, self).__init__(n_inputs, n_outputs, intput_dtype, output_dtype, seed)
        self.name = 'Regression neural network'
        self.output_activator = None
        self.l2 = l2

    def _init_loss(self):
        with tf.name_scope('loss') as scope:
            self.loss = tf.reduce_mean(tf.reduce_sum(
                tf.square(self.target_placeholder - self.layers_output.iloc[-1], name='loss_square'),
                reduction_indices=[1]), name='loss_mean')
            regularizers = 0
            for W in self.Ws:
                regularizers += tf.nn.l2_loss(W, name='l2_reg')
            self.loss += self.l2*regularizers


