import time
from sklearn.datasets import fetch_olivetti_faces
from sklearn.datasets import load_boston
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import tfnn
import numpy as np

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

network = tfnn.RegressionNetwork(4096, 40, dtype=tf.float32)
# network.add_hidden_layer(100, activator=tf.nn.relu)
optimizer = tf.train.GradientDescentOptimizer(0.0001)
network.set_optimizer(optimizer)

for i in range(2000):
    b_xs, b_ys = t_data.next_batch(50, loop=True)
    # b_xs, b_ys = mnist.train.next_batch(100)
    network.run_step(b_xs, b_ys)
    if i % 100 == 0:
        loss_value = network.get_loss(v_data.xs, v_data.ys)
        print(loss_value)
# evalu = tfnn.compute_accuracy(network, mnist.test.images, mnist.test.labels)
# print(evalu)
