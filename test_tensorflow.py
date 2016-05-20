import time
from sklearn.datasets import fetch_olivetti_faces
from sklearn.datasets import load_boston
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import tfnn
import numpy as np

# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

tf.set_random_seed(100)

xs = load_boston().data
ys = load_boston().target
# xs = fetch_olivetti_faces().data
# ys = fetch_olivetti_faces().target[:, np.newaxis]
data = tfnn.Data(xs, ys)
data.minmax_normalize(inplace=True)
data.shuffle(inplace=True)
# data.to_binary(inplace=True)
t_data, v_data = data.train_test_split()
network = tfnn.RegressionNetwork(data.xs.shape[1], data.ys.shape[1], do_dropout=True)
# network.add_hidden_layer(100, activator=tf.nn.relu)
# network.add_hidden_layer(10, activator=tf.nn.tanh)
network.add_output_layer(activator=None)
optimizer = tf.train.GradientDescentOptimizer(0.01)
network.set_optimizer(optimizer)
evaluator = tfnn.AccuracyEvaluator(network)
summarizer = tfnn.Summarizer(network, save_path='/tmp/log')

for i in range(1000):
    b_xs, b_ys = t_data.next_batch(50, loop=True)
    # b_xs, b_ys = mnist.train.next_batch(100)
    network.run_step(b_xs, b_ys, 0.5)
    # if i % 100 == 0:
    #     loss_value = network.get_loss(v_data.xs, v_data.ys)
    #     print(loss_value)
    if i % 20 == 0:
        # print(evaluator.compute_accuracy(v_data.xs, v_data.ys))
        # evaluator.plot_single_output_comparison(v_data.xs, v_data.ys, True)
        summarizer.record_train(b_xs, b_ys, i, 0.5)
        summarizer.record_validate(v_data.xs, v_data.ys, i)

network.sess.close()

