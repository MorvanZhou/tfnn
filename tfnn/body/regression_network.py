from tfnn.body.network import Network
import tensorflow as tf


class RegressionNetwork(Network):
    def __init__(self, n_inputs, n_outputs, intput_dtype=tf.float32, output_dtype=tf.float32,
                 do_dropout=False, do_l2=False, seed=None):
        output_activator = None
        super(RegressionNetwork, self).__init__(
            n_inputs, n_outputs, intput_dtype, output_dtype, output_activator,
            do_dropout, do_l2, seed)
        self.name = 'RegressionNetwork'

    def _init_loss(self):
        with tf.name_scope('predictions'):
            self.predictions = self.layers_final_output.iloc[-1] + 0
        with tf.name_scope('loss'):
            loss_square = tf.square(self.target_placeholder - self.layers_output.iloc[-1], name='loss_square')
            loss_sum = tf.reduce_sum(loss_square, reduction_indices=[1], name='loss_sum')
            self.loss = tf.reduce_mean(loss_sum, name='loss_mean')

            if self.reg == 'l2':
                with tf.name_scope('l2_reg'):
                    regularizers = 0
                    for W in self.Ws:
                        regularizers += tf.nn.l2_loss(W, name='l2_reg')
                    regularizers *= self.l2_placeholder
                with tf.name_scope('l2_loss'):
                    self.loss += regularizers
            tf.scalar_summary('loss', self.loss)


