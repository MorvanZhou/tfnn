import tensorflow as tf


def compute_accuracy(network, xs, ys):
    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(network.layers_output.iloc[-1], 1),
                                      tf.argmax(network.target_placeholder, 1), name='correct_prediction')
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
    return network.sess.run(accuracy, feed_dict={network.data_placeholder: xs,
                                                 network.target_placeholder: ys})
