from tfnn.body.network import Network
import tensorflow as tf


class ClassificationNetwork(Network):
    def __init__(self, n_inputs, n_outputs, method='softmax', l2=0, seed=None):
        """

        :param n_inputs:
        :param n_outputs:
        :param method: 'sparse_softmax' or 'softmax' or 'sigmoid'
        :param dtype:
        :param seed:
        """
        if method == 'softmax':
            input_dtype = tf.float32
            output_dtype = tf.float32
            self.output_activator = tf.nn.softmax
        elif method == 'sparse_softmax':
            input_dtype = tf.float32
            output_dtype = tf.int32
            self.output_activator = tf.nn.softmax
        elif method == 'sigmoid':
            input_dtype = tf.float32
            output_dtype = tf.float32
            self.output_activator = tf.sigmoid
        else:
            raise ValueError("method should be one of ['sparse_softmax', 'softmax', 'sigmoid']")
        super(ClassificationNetwork, self).__init__(n_inputs, n_outputs, input_dtype, output_dtype, seed)
        self.method = method
        self.l2 = l2
        self.name = 'Classification neural network'

    def _init_loss(self):
        with tf.name_scope('loss'):
            with tf.name_scope('xentropy'):
                if self.method == 'softmax':
                    self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                        self.layers_output.iloc[-1],
                        self.target_placeholder,
                        name='xentropy')
                elif self.method == 'sparse_softmax':
                    self.cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                        self.layers_output.iloc[-1],
                        self.target_placeholder,
                        name='xentropy')
                elif self.method == 'sigmoid':
                    self.cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
                        self.layers_output.iloc[-1],
                        self.target_placeholder,
                        name='xentropy')
                else:
                    raise ValueError("method should be one of ['sparse_softmax', 'softmax', 'sigmoid']")
                self.loss = tf.reduce_mean(self.cross_entropy, name='xentropy_mean')

            with tf.name_scope('l2_reg'):
                regularizers = 0
                for W in self.Ws:
                    regularizers += tf.nn.l2_loss(W, name='l2_reg')
                regularizers *= self.l2
            with tf.name_scope('l2_loss'):
                self.loss += regularizers
