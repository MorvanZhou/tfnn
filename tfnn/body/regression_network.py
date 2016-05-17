from tfnn.body.network import Network
import tensorflow as tf


class RegressionNetwork(Network):
    def __init__(self, n_inputs, n_outputs, intput_dtype=tf.float32, output_dtype=tf.float32, seed=None):
        super(RegressionNetwork, self).__init__(n_inputs, n_outputs, intput_dtype, output_dtype, seed)
        self.name = 'Regression neural network'

    def _init_loss(self):
        self.loss = tf.reduce_mean(tf.reduce_sum(
            tf.square(self.target_placeholder - self.layers_output.iloc[-1], name='loss_square'),
            reduction_indices=[1]), name='loss_mean')


