from tfnn.body.network import Network
import tfnn
import numpy as np


class RegNetwork(Network):
    def __init__(self, n_inputs, n_outputs, intput_dtype=tfnn.float32, output_dtype=tfnn.float32,
                 do_dropout=False, do_l2=False, seed=None):
        """

        :param input_size: if CNN, this is the image [length, width, channels]
        :param output_size:
        :param intput_dtype:
        :param output_dtype:
        :param do_dropout:
        :param do_l2:
        :param seed:
        """
        output_activator = None
        super(RegNetwork, self).__init__(
            n_inputs, n_outputs, intput_dtype, output_dtype, output_activator,
            do_dropout, do_l2, seed)
        self.name = 'RegressionNetwork'

    def _init_loss(self):
        with tfnn.name_scope('predictions'):
            self.predictions = self.layers_results['final'].iloc[-1] + 0
        with tfnn.name_scope('loss'):
            loss_square = tfnn.square(self.target_placeholder - self.predictions,
                                      name='loss_square')
            loss_sum = tfnn.reduce_sum(loss_square, reduction_indices=[1], name='loss_sum')
            self.loss = tfnn.reduce_mean(loss_sum, name='loss_mean')

            if self.reg == 'l2':
                with tfnn.name_scope('l2_reg'):
                    regularizers = 0
                    for W in self.Ws:
                        regularizers += tfnn.nn.l2_loss(W, name='l2_reg')
                    regularizers *= self.l2_placeholder
                with tfnn.name_scope('l2_loss'):
                    self.loss += regularizers
            tfnn.scalar_summary('loss', self.loss)

    def predict(self, xs):
        if np.ndim(xs) == 1:
            xs = xs[np.newaxis, :]
        predictions = self.sess.run(self.predictions, feed_dict={self.data_placeholder: xs})
        if predictions.size == 1:
            predictions = predictions[0][0]
        return predictions
