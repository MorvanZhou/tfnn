import time
from sklearn.datasets import fetch_olivetti_faces
from sklearn.datasets import load_boston
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import tfnn
import numpy as np

# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

tfnn.set_random_seed(100)

xs = load_iris().data
ys = load_iris().target
# xs = fetch_olivetti_faces().data
# ys = fetch_olivetti_faces().target[:, np.newaxis]

# xs = np.linspace(-5, 5, 100000)[:, np.newaxis]
# ys = xs**2 + np.random.normal(size=xs.shape)

data = tfnn.Data(xs, ys)
data.shuffle(inplace=True)
data.onehot_encode_y(True)

# network = tfnn.ClfNetwork(mnist.train.images.shape[1], mnist.train.labels.shape[1], do_dropout=False)
network = tfnn.ClfNetwork(data.xs.shape[1], data.ys.shape[1], )
# network = tfnn.RegNetwork(data.xs.shape[1], data.ys.shape[1], do_dropout=True)
norm_data = network.normalizer.minmax(data)
t_data, v_data = norm_data.train_test_split(0.7)

h1 = tfnn.HiddenLayer(10, activator='relu', dropout_layer=True)
h2 = tfnn.HiddenLayer(10, activator='tanh', dropout_layer=False)
out = tfnn.OutputLayer(activator=None, w_initial='random_normal')
network.build_layers([h1, h2, out])

network.set_optimizer('GD')
network.set_learning_rate(lr=0.01, exp_decay=dict(decay_steps=1000, decay_rate=0.9, staircase=False))

evaluator = tfnn.Evaluator(network)
# evaluator.set_line_fitting_monitor()
# evaluator.set_data_fitting_monitor()
evaluator.set_scale_monitor(['cost', 'accuracy', 'learning rate', 'f1', 'recall'], figsize=(7, 7))
evaluator.set_layer_monitor([0, 1, 2], cbar_range=(-0.4, 0.4))

# write summarizer at the end of the structure
# summarizer = tfnn.Summarizer(network, save_path='tmp',)

st = time.time()
for i in range(2000):
    b_xs, b_ys = t_data.next_batch(50)
    # b_xs, b_ys = mnist.train.next_batch(50)
    network.run_step(b_xs, b_ys, keep_prob=0.5)
    if i % 100 == 0:
        now = time.time()
        print('time spend: ', round(now - st, 2))
        st = now
        # print(evaluator.compute_accuracy(v_data.xs, v_data.ys))
        # print(evaluator.compute_cost(v_data.xs, v_data.ys))
        # print(evaluator.compute_accuracy(b_xs, b_ys))
        evaluator.monitoring(b_xs, b_ys, v_xs=v_data.xs, v_ys=v_data.ys,)
        # evaluator.monitoring(b_xs, b_ys, v_xs=mnist.test.images, v_ys=mnist.test.labels)
        # summarizer.record_train(b_xs, b_ys,)
        # summarizer.record_test(v_data.xs, v_data.ys)
        # summarizer.record_test(mnist.test.images, mnist.test.labels, i)
evaluator.hold_plot()
# summarizer.web_visualize()
network.sess.close()


