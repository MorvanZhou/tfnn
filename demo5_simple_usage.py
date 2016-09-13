from sklearn.datasets import load_boston
import tfnn

# load data
xs = load_boston().data
ys = load_boston().target

# set data into tfnn.Data format
data = tfnn.Data(xs, ys)
data.shuffle(inplace=True)

# define network properties
network = tfnn.RegNetwork(input_size=data.xs.shape[1], output_size=data.ys.shape[1])

# normalize features
norm_data = network.normalizer.minmax(data)

# train test split
t_data, v_data = norm_data.train_test_split(0.7)

# set hidden layer
h1 = tfnn.HiddenLayer(n_neurons=10, activator='relu')
h2 = tfnn.HiddenLayer(n_neurons=10, activator='relu')

# set output layer
out = tfnn.OutputLayer(activator=None)

# build network layers
network.build_layers([h1, h2, out])

# set optimizer. Default GradientDescent
network.set_optimizer('GD')

# set learning rate
network.set_learning_rate(0.0001)

# set evaluator for compute the accuracy, loss etc.
evaluator = tfnn.Evaluator(network)

# set Score Monitor for live plot
evaluator.set_scale_monitor(['r2', 'cost'])

# train network
for step in range(400):
    b_xs, b_ys = t_data.next_batch(20,)
    network.run_step(b_xs, b_ys)
    if step % 10 == 0:
        evaluator.monitoring(b_xs, b_ys, v_xs=v_data.xs, v_ys=v_data.ys)
evaluator.hold_plot()



