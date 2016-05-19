import time
from sklearn.datasets import fetch_olivetti_faces
from sklearn.datasets import load_boston
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import tfnn
import numpy as np
import os, shutil

# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

tf.set_random_seed(100)

# xs = load_boston().data
# ys = load_boston().target
xs = fetch_olivetti_faces().data
ys = fetch_olivetti_faces().target[:, np.newaxis]
data = tfnn.Data(xs, ys)
data.minmax_normalize(inplace=True)
data.shuffle(inplace=True)
data.to_binary(inplace=True)
t_data, v_data = data.train_test_split()
network = tfnn.ClassificationNetwork(4096, 40, do_dropout=True)
# network.add_hidden_layer(100, activator=tf.nn.relu)
# network.add_hidden_layer(10, activator=tf.nn.tanh)
network.add_output_layer(activator=None)
optimizer = tf.train.GradientDescentOptimizer(0.001)
network.set_optimizer(optimizer)
evaluator = tfnn.AccuracyEvaluator(network, v_data.xs, v_data.ys)
graph_save_path = '/tmp/log'
if 'log' in os.listdir('/tmp'):
    shutil.rmtree(graph_save_path)
merged = tf.merge_all_summaries()
train_writer = tf.train.SummaryWriter(graph_save_path + '/train', network.sess.graph)
test_writer = tf.train.SummaryWriter(graph_save_path + '/test', )

for i in range(1000):
    b_xs, b_ys = t_data.next_batch(50, loop=True)
    # b_xs, b_ys = mnist.train.next_batch(100)
    network.run_step(b_xs, b_ys, 0.5)
    # if i % 100 == 0:
    #     loss_value = network.get_loss(v_data.xs, v_data.ys)
    #     print(loss_value)
    if i%50 == 0:
        test_result = network.sess.run(merged, feed_dict={network.data_placeholder: v_data.xs,
                                                          network.target_placeholder: v_data.ys,
                                                          network.keep_prob_placeholder: 0.5})
        train_result = network.sess.run(merged, feed_dict={network.data_placeholder: b_xs,
                                                          network.target_placeholder: b_ys,
                                                          network.keep_prob_placeholder: 1})
        test_writer.add_summary(test_result, i)
        train_writer.add_summary(train_result, i)

network.sess.close()

