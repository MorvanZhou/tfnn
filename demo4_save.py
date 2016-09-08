from tensorflow.examples.tutorials.mnist import input_data
import tfnn

# digits image data from 0 to 9
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# define network properties
network = tfnn.ClfNetwork(input_size=mnist.train.images.shape[1], output_size=mnist.train.labels.shape[1])

# add hidden layer
network.add_hidden_layer(n_neurons=100, activator=tfnn.nn.relu)

# add output layer
network.add_output_layer(activator=None)

# set optimizer. Default GradientDescent
network.set_optimizer()

# set evaluator for compute the accuracy, loss etc.
evaluator = tfnn.Evaluator(network)

# similar to sklearn, we have fit function
network.fit(mnist.train.images, mnist.train.labels, steps=2000)
print('trained network predict:')
print(network.predict(mnist.test.images[10:20]))
print('real data value:')
print(mnist.test.labels[10:20].argmax(axis=1))

# save network
network.save(name='model', path='tmp')

# reload network
saver = tfnn.NetworkSaver()
network2 = saver.restore(name='model', path='tmp')
print('\nLoaded network predict:')
print(network2.predict(mnist.test.images[10:20]))


