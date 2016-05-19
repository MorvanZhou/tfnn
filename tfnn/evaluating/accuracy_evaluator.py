import tensorflow as tf


class AccuracyEvaluator(object):
    def __init__(self, network, xs, ys):
        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(tf.argmax(network.predictions, 1),
                                          tf.argmax(network.target_placeholder, 1), name='correct_prediction')
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
            tf.scalar_summary('accuracy', self.accuracy)
        if network.reg == 'dropout':
            feed_dict = {network.data_placeholder: xs,
                         network.target_placeholder: ys,
                         network.keep_prob_placeholder: 1.}
        elif network.reg == 'l2':
            feed_dict = {network.data_placeholder: xs,
                         network.target_placeholder: ys,
                         network.l2_placeholder: 0.}
        else:
            feed_dict = {network.data_placeholder: xs,
                         network.target_placeholder: ys}
        self.feed_dict = feed_dict
        self.network = network

    def compute_accuracy(self):
        return self.network.sess.run(self.accuracy, self.feed_dict)
