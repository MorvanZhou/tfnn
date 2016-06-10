import tfnn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def train():
    load_data = pd.read_csv('1s data.csv')
    xs = load_data.iloc[:, 8:]
    ys = load_data.a
    data = tfnn.Data(xs, ys, name='road_data')
    data.minmax_normalize(inplace=True)
    t_data, v_data = data.train_test_split(0.7)

    network = tfnn.RegressionNetwork(xs.shape[1], 1, do_dropout=True)
    network.add_hidden_layer(100, activator=tfnn.nn.relu, dropout_layer=True)
    network.add_output_layer(activator=None, dropout_layer=False)
    global_step = tfnn.Variable(0, trainable=False)
    # lr = tfnn.train.exponential_decay(0.001, global_step, 2000, 0.9)
    optimizer = tfnn.train.AdamOptimizer(0.01)
    network.set_optimizer(optimizer, global_step)
    evaluator = tfnn.Evaluator(network)
    summarizer = tfnn.Summarizer(save_path='/tmp/log', network=network)

    for i in range(15000):
        b_xs, b_ys = t_data.next_batch(300, loop=True)
        network.run_step(b_xs, b_ys, 0.5)
        if i % 500 == 0:
            print(evaluator.compute_cost(v_data.xs, v_data.ys))
            summarizer.record_train(b_xs, b_ys, i, 0.5)
            summarizer.record_validate(v_data.xs, v_data.ys, i)
    network_saver = tfnn.NetworkSaver()
    network_saver.save(network, data)
    evaluator.plot_single_output_comparison(v_data.xs, v_data.ys, False)
    summarizer.web_visualize()
    network.sess.close()


def test():
    load_data = pd.read_csv('1s data.csv')
    xs = load_data.iloc[6300:6600, 8:]
    ys = load_data.a[6300:6600][:, np.newaxis]
    network_saver = tfnn.NetworkSaver()
    restore_path = '/tmp/'
    network, input_filter = network_saver.rebuild(restore_path)
    network_saver.restore(restore_path)
    prediction = network.predict(input_filter.filter(xs))
    plt.plot(np.arange(xs.shape[0]), prediction, 'r-', label='predicted')
    plt.plot(np.arange(xs.shape[0]), ys, 'k-', label='real')
    plt.legend(loc='best')
    plt.show()
    network.sess.close()


if __name__ == '__main__':
    test()