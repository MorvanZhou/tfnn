import tfnn
from tensorflow.examples.tutorials.mnist import input_data

# this is data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
xs = mnist.train.images
ys = mnist.train.labels
data = tfnn.Data(xs, ys)
train_data, test_data = data.train_test_split(train_rate=0.9)

# to select one classification network
network = tfnn.ClfNetwork(xs.shape[1], ys.shape[1], do_dropout=True)

# convolution layer1
network.add_conv_layer(patch_x=5, patch_y=5, n_filters=32,
                       activator=tfnn.nn.relu, image_shape=(28, 28, 1))
# convolution layer2
network.add_conv_layer(patch_x=5, patch_y=5, n_filters=64,
                       activator=tfnn.nn.relu)
# fully connected layer1
network.add_fc_layer(1024, tfnn.nn.relu, dropout_layer=True)

# fully connected output layer
network.add_output_layer()

# choose optimizer
network.set_optimizer(optimizer='adam')
network.set_learning_rate(0.01)

# set evaluator
evaluator = tfnn.Evaluator(network)

# set summarizer at the end of neural structure, and visualize results in tensorboard
summarizer = tfnn.Summarizer(network, save_path='/tmp',)

for i in range(200):
    b_xs, b_ys = train_data.next_batch(100)
    # train with keep probability of 0.5
    network.run_step(b_xs, b_ys, 0.5)
    if i % 30 == 0:
        # record
        print('accuracy:', evaluator.compute_accuracy(b_xs, b_ys))
        summarizer.record_train(b_xs, b_ys)
# visualize it on tensorborad
summarizer.web_visualize()




