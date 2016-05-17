from tfnn.body.network import Network
import tensorflow as tf


class ClassificationNetwork(Network):
    def __init__(self, n_inputs, n_outputs, dtype=tf.float32, seed=None):
        super(ClassificationNetwork, self).__init__(n_inputs, n_outputs, dtype, seed)
        self.name = 'Classification neural network'

    def _init_loss(self):
        self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            self.layers_output.iloc[-1],
            self.target_placeholder,
            name='xentropy')
        self.loss = tf.reduce_mean(self.cross_entropy, name='xentropy_mean')