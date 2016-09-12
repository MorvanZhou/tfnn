import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

# this is data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# hyperparameters
lr = 0.001
training_iters = 100000
batch_size = 128
display_step = 10

n_inputs = 28   # MNIST data input (img shape: 28*28)
n_steps = 28    # time steps
n_hidden_unis = 128   # neurons in hidden layer
n_classes = 10      # MNIST classes (0-9 digits)

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
def length(data):
    used = tf.sign(tf.reduce_max(tf.abs(data), reduction_indices=2))
    length = tf.reduce_sum(used, reduction_indices=1)
    length = tf.cast(length, tf.int32)
    return length

# for LSTM the init_state include units for gates and units for hidden. so 2*hidden_units as default
init_state = tf.placeholder(tf.float32, [None, 2*n_hidden_unis])
y = tf.placeholder(tf.float32, [None, n_classes])

# Define weights
weights = {
    # (28, 128)
    'hidden': tf.Variable(tf.random_normal([n_inputs, n_hidden_unis])),
    # (128, 10)
    'out': tf.Variable(tf.random_normal([n_hidden_unis, n_classes]))
}
biases = {
    # (128, )
    'hidden': tf.Variable(tf.constant(0.1, shape=[n_hidden_unis, ])),
    # (10, )
    'out': tf.Variable(tf.constant(0.1, shape=[n_classes, ]))
}


def RNN(X, init_state, weights, biases):
    # transpose the inputs shape from
    # (batch_size=128, n_steps=28, n_inputs=28) = (128, 28, 28)
    # to (n_steps, batch_size, n_inputs) = (28, 128, 28)
    X = tf.transpose(X, [1, 0, 2])
    # reshape to be accepted by
    # hidden layer (28, 128), so the X is (-1, 28)
    X = tf.reshape(X, [-1, n_inputs])

    # into hidden
    # (28*128, 28) dot (28, 128) + (128, ) ==> (28*128, 128)
    X = tf.matmul(X, weights['hidden']) + biases['hidden']
    # (28*128, 128)==> steps=28 * (batch=128, hidden=128)
    X = tf.split(0, n_steps, X)
    # ==> a list of X. [X1, X2, ..., X28] 28 steps
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_unis, forget_bias=1.0)
    outputs, states = tf.nn.rnn(lstm_cell, X, initial_state=init_state)
    # return final outputs
    return tf.matmul(outputs[-1], weights['out']) + biases['out']


pred = RNN(x, init_state, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
train_op = tf.train.AdamOptimizer(lr).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)
    step = 1
    while step * batch_size < training_iters:
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
        sess.run(train_op, feed_dict={
            x: batch_xs,
            y: batch_ys,
            init_state: np.zeros((batch_size, 2*n_hidden_unis))
        })





