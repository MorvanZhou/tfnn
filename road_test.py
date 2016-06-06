import tfnn
import pandas as pd

load_data = pd.read_csv('tmp2.csv')
xs = load_data[['v', 's', 'v_f']]
ys = load_data.a
data = tfnn.Data(xs, ys, name='road_data')
data.minmax_normalize(inplace=True)

t_data, v_data = data.train_test_split(0.1)
network = tfnn.RegressionNetwork(xs.shape[1], 1, do_dropout=False)
network.add_hidden_layer(10, activator=tfnn.nn.relu)
network.add_output_layer(activator=None)
global_step = tfnn.Variable(0, trainable=False)
optimizer = tfnn.train.GradientDescentOptimizer(0.01)
network.set_optimizer(optimizer, global_step)
evaluator = tfnn.Evaluator(network)
summarizer = tfnn.Summarizer(network, save_path='/tmp/log')

for i in range(20000):
    b_xs, b_ys = t_data.next_batch(100, loop=True)
    network.run_step(b_xs, b_ys, 0.5)
    if i % 100 == 0:

        print(evaluator.compute_cost(v_data.xs, v_data.ys))
        summarizer.record_train(b_xs, b_ys, i, 0.5)
        summarizer.record_validate(v_data.xs, v_data.ys, i)
evaluator.plot_single_output_comparison(v_data.xs, v_data.ys, True)
summarizer.web_visualize()
network.sess.close()
