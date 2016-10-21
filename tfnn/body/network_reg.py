from tfnn.body.network import Network
import tfnn
import numpy as np


class RegNetwork(Network):
    def __init__(self, input_size, output_size, do_dropout=False, do_l2=False):

        super(RegNetwork, self).__init__(
            input_size, output_size, do_dropout, do_l2, ntype='RNet')
        self.name = 'RegressionNetwork'
        self._params = {
            'input_size': input_size,
            'output_size': output_size,
            'do_dropout': do_dropout,
            'do_l2': do_l2,
        }
        self.layers_configs['params'] = [self._params]

    def __str__(self):
        return self.name

    def _init_loss(self):
        with tfnn.name_scope('predictions'):
            self.predictions = self.layers_results['final'][-1]
        with tfnn.name_scope('loss'):
            loss_square = tfnn.square(self.target_placeholder - self.predictions,
                                      name='loss_square')
            loss_sum = tfnn.reduce_sum(loss_square, reduction_indices=[1], name='loss_sum')
            self.loss = tfnn.reduce_mean(loss_sum, name='loss_mean')

            if self.reg == 'l2':
                with tfnn.name_scope('l2_reg'):
                    regularizers = 0
                    for layer in self.layers_results['Layer'][1:]:
                        regularizers += tfnn.nn.l2_loss(layer.W, name='l2_reg')
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
