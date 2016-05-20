from tfnn.body.network import Network
import tensorflow as tf


class ClassificationNetwork(Network):
    def __init__(self, n_inputs, n_outputs, method='softmax', do_dropout=False, do_l2=False, seed=None):
        if method == 'softmax':
            input_dtype = tf.float32
            output_dtype = tf.float32
            output_activator = tf.nn.softmax
        elif method == 'sigmoid':
            input_dtype = tf.float32
            output_dtype = tf.float32
            output_activator = tf.sigmoid
        else:
            raise ValueError("method should be one of ['softmax', 'sigmoid']")
        super(ClassificationNetwork, self).__init__(
            n_inputs, n_outputs, input_dtype, output_dtype, output_activator,
            do_dropout, do_l2, seed)
        self.method = method
        self.name = 'ClassificationNetwork'

    def _init_loss(self):
        if self.method == 'softmax':
            self.predictions = tf.nn.softmax(self.layers_final_output.iloc[-1], name='predictions')
        elif self.method == 'sigmoid':
            self.predictions = tf.nn.sigmoid(self.layers_final_output.iloc[-1], name='predictions')
        with tf.name_scope('loss'):
            if self.method == 'softmax':
                self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                    self.layers_final_output.iloc[-1],
                    self.target_placeholder,
                    name='xentropy')
            elif self.method == 'sigmoid':
                self.cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
                    self.layers_final_output.iloc[-1],
                    self.target_placeholder,
                    name='xentropy')
            else:
                raise ValueError("method should be one of ['sparse_softmax', 'softmax', 'sigmoid']")
            self.loss = tf.reduce_mean(self.cross_entropy, name='xentropy_mean')

            if self.reg == 'l2':
                with tf.name_scope('l2_reg'):
                    regularizers = 0
                    for W in self.Ws:
                        regularizers += tf.nn.l2_loss(W, name='l2_reg')
                    regularizers *= self.l2_placeholder
                with tf.name_scope('l2_loss'):
                    self.loss += regularizers

            tf.scalar_summary('loss', self.loss)
