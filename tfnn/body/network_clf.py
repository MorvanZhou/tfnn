from tfnn.body.network import Network
import tfnn
import numpy as np


class ClfNetwork(Network):
    def __init__(self, input_size, output_size, method='softmax', do_dropout=False, do_l2=False,):

        if method not in ['softmax', 'sigmoid']:
            raise ValueError("method should be one of ['softmax', 'sigmoid']")
        super(ClfNetwork, self).__init__(
            input_size, output_size, do_dropout, do_l2, ntype='CNet')
        self.method = method
        self.name = 'ClassificationNetwork'
        self._params = {
            'input_size': input_size,
            'output_size': output_size,
            'method': method,
            'do_dropout': do_dropout,
            'do_l2': do_l2,
        }
        self.layers_configs['params'] = [self._params]

    def __str__(self):
        return self.name

    def _init_loss(self):
        with tfnn.name_scope('predictions'):
            if self.method == 'softmax':
                self.predictions = tfnn.nn.softmax(self.layers_results['final'][-1], name='predictions')
            elif self.method == 'sigmoid':
                self.predictions = tfnn.nn.sigmoid(self.layers_results['final'][-1], name='predictions')
        with tfnn.name_scope('loss'):
            if self.method == 'softmax':
                self.cross_entropy = tfnn.nn.softmax_cross_entropy_with_logits(
                    self.layers_results['final'][-1],
                    self.target_placeholder,
                    name='xentropy')
            elif self.method == 'sigmoid':
                self.cross_entropy = tfnn.nn.sigmoid_cross_entropy_with_logits(
                    self.layers_results['final'][-1],
                    self.target_placeholder,
                    name='xentropy')
            else:
                raise ValueError("method should be one of ['sparse_softmax', 'softmax', 'sigmoid']")
            self.loss = tfnn.reduce_mean(self.cross_entropy, name='xentropy_mean')

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
        predictions = np.argmax(predictions, axis=1)
        if predictions.size == 1:
            predictions = predictions[0][0]
        return predictions

    def predict_prob(self, xs):
        if np.ndim(xs) == 1:
            xs = xs[np.newaxis, :]
        predictions = self.sess.run(self.predictions, feed_dict={self.data_placeholder: xs})
        if predictions.size == 1:
            predictions = predictions[0][0]
        return predictions