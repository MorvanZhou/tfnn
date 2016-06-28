from tensorflow.examples.tutorials.mnist import input_data
import tfnn

# digits image data from 0 to 9
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# define network properties
network = tfnn.ClassificationNetwork(n_inputs=mnist.train.images.shape[1], n_outputs=mnist.train.labels.shape[1])

# add hidden layer
network.add_hidden_layer(n_neurons=20, activator=tfnn.nn.relu)

# add output layer
network.add_output_layer(activator=None)

# set optimizer. Default GradientDescent
network.set_optimizer()

# set evaluator for compute the accuracy, loss etc.
evaluator = tfnn.Evaluator(network)

for i in range(3000):
    # batch data
    batch_xs, batch_ys = mnist.train.next_batch(100)

    # learning from batch data
    network.run_step(feed_xs=batch_xs, feed_ys=batch_ys)

    if i % 50 == 0:
        # print predict accuracy
        print(evaluator.compute_accuracy(xs=mnist.test.images, ys=mnist.test.labels))
