import tfnn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def train():
    load_data = pd.read_csv('tmp2.csv')
    xs = load_data.iloc[:, -13:]
    ys = load_data.a
    data = tfnn.Data(xs, ys, name='road_data')
    data.minmax_normalize(inplace=True)
    t_data, v_data = data.train_test_split(0.7)

    network = tfnn.RegressionNetwork(xs.shape[1], 1, do_dropout=False)
    network.add_hidden_layer(40, activator=tfnn.nn.relu)
    network.add_output_layer(activator=None)
    global_step = tfnn.Variable(0, trainable=False)
    optimizer = tfnn.train.GradientDescentOptimizer(0.001)
    network.set_optimizer(optimizer, global_step)
    evaluator = tfnn.Evaluator(network)
    # summarizer = tfnn.Summarizer(network, save_path='/tmp/log')

    for i in range(10000):
        b_xs, b_ys = t_data.next_batch(100, loop=True)
        network.run_step(b_xs, b_ys, 0.5)
        if i % 1000 == 0:
            print(evaluator.compute_cost(v_data.xs, v_data.ys))
            # summarizer.record_train(b_xs, b_ys, i, 0.5)
            # summarizer.record_validate(v_data.xs, v_data.ys, i)
    network_saver = tfnn.NetworkSaver()
    network_saver.save(network, data)
    evaluator.plot_single_output_comparison(v_data.xs, v_data.ys, False)
    # summarizer.web_visualize()
    network.sess.close()


def test():
    load_data = pd.read_csv('tmp2.csv')
    xs = load_data.iloc[35000:35500, -13:]
    ys = load_data.a[35000:35500][:, np.newaxis]
    network_saver = tfnn.NetworkSaver()
    restore_path = '/tmp'
    network, input_filter = network_saver.rebuild(restore_path)
    network_saver.restore(restore_path)
    prediction = network.predict(input_filter.filter(xs))
    plt.plot(np.arange(xs.shape[0]), prediction, 'r-')
    plt.plot(np.arange(xs.shape[0]), ys, 'k-')
    plt.show()
    network.sess.close()


if __name__ == '__main__':
    test()