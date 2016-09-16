import tensorflow as tf
import matplotlib.pyplot as plt
from numpy import array, sin, cos, pi
from random import random

# Random initial angles
angle1 = random()
angle2 = random()

# The total 2*pi cycle would be divided into 'frequency'
# number of steps
frequency1 = 300
frequency2 = 200
# This defines how many steps ahead we are trying to predict
lag = 23


def get_sample():
    """
    Returns a [[sin value, cos value]] input.
    """
    global angle1, angle2
    angle1 += 2 * pi / float(frequency1)
    angle2 += 2 * pi / float(frequency2)
    angle1 %= 2 * pi
    angle2 %= 2 * pi
    return array([array([
        5 + 5 * sin(angle1) + 10 * cos(angle2),
        7 + 7 * sin(angle2) + 14 * cos(angle1)])])


sliding_window = []

for i in range(lag - 1):
    sliding_window.append(get_sample())


def get_pair():
    """
    Returns an (current, later) pair, where 'later' is 'lag'
    steps ahead of the 'current' on the wave(s) as defined by the
    frequency.
    """

    global sliding_window
    sliding_window.append(get_sample())
    input_value = sliding_window[0]
    output_value = sliding_window[-1]
    sliding_window = sliding_window[1:]
    return input_value, output_value


# Input Params
input_dim = 2

# To maintain state
last_value = array([0 for i in range(input_dim)])
last_derivative = array([0 for i in range(input_dim)])


def get_total_input_output():
    """
    Returns the overall Input and Output as required by the model.
    The input is a concatenation of the wave values, their first and
    second derivatives.
    """
    global last_value, last_derivative
    raw_i, raw_o = get_pair()
    raw_i = raw_i[0]
    l1 = list(raw_i)
    derivative = raw_i - last_value
    l2 = list(derivative)
    last_value = raw_i
    l3 = list(derivative - last_derivative)
    last_derivative = derivative
    return array([l1 + l2 + l3]), raw_o

# Input Params
input_dim = 2

##The Input Layer as a Placeholder
# Since we will provide data sequentially, the 'batch size'
# is 1.
input_layer = tf.placeholder(tf.float32, [1, input_dim * 3])
correct_output = tf.placeholder(tf.float32, [1, input_dim])

lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(input_dim*3)
lstm_state = lstm_cell.zero_state(1, tf.float32)
lstm_output, lstm_state_new = lstm_cell(input_layer, lstm_state)

lstm_state = lstm_state_new

Wo = tf.Variable(tf.truncated_normal([input_dim*3, input_dim]))
bo = tf.Variable(tf.zeros([input_dim]))
final_output = tf.matmul(lstm_output, Wo) + bo

error = tf.pow(tf.sub(final_output, correct_output), 2)
train_step = tf.train.AdamOptimizer(0.0006).minimize(error)

sess = tf.Session()
sess.run(tf.initialize_all_variables())

actual_output1 = []
actual_output2 = []
network_output1 = []
network_output2 = []
x_axis = []

for i in range(10000):
    input_v, output_v = get_total_input_output()
    _, network_output = sess.run([
                                     train_step,
                                     final_output],
                                    feed_dict={
                                        input_layer: input_v,
                                        correct_output: output_v})

    actual_output1.append(output_v[0][0])
    actual_output2.append(output_v[0][1])
    network_output1.append(network_output[0][0])
    network_output2.append(network_output[0][1])
    x_axis.append(i)


plt.plot(x_axis, network_output1, 'r-', x_axis, actual_output1, 'b-')
plt.show()
plt.plot(x_axis, network_output2, 'r-', x_axis, actual_output2, 'b-')
plt.show()