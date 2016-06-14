import tfnn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def train(data_path):
    load_data = pd.read_pickle(data_path)
    xs = load_data.iloc[:, 1:]
    print(xs.head(2))
    ys = load_data.a
    data = tfnn.Data(xs, ys, name='road_data')
    data.minmax_normalize(inplace=True)
    t_data, v_data = data.train_test_split(0.7)

    network = tfnn.RegressionNetwork(xs.shape[1], 1, do_dropout=True)
    network.add_hidden_layer(200, activator=tfnn.nn.relu, dropout_layer=True)
    network.add_output_layer(activator=None, dropout_layer=False)
    global_step = tfnn.Variable(0, trainable=False)
    # lr = tfnn.train.exponential_decay(0.001, global_step, 2000, 0.9)
    optimizer = tfnn.train.AdamOptimizer(0.001)
    network.set_optimizer(optimizer, global_step)
    evaluator = tfnn.Evaluator(network)
    summarizer = tfnn.Summarizer(save_path='/tmp/log', network=network)

    for i in range(20000):
        b_xs, b_ys = t_data.next_batch(100, loop=True)
        network.run_step(b_xs, b_ys, 0.5)
        if i % 1000 == 0:
            print(evaluator.compute_cost(v_data.xs, v_data.ys))
            summarizer.record_train(b_xs, b_ys, i, 0.5)
            summarizer.record_validate(v_data.xs, v_data.ys, i)
    # network_saver = tfnn.NetworkSaver()
    # network_saver.save(network, data)
    evaluator.plot_single_output_comparison(v_data.xs, v_data.ys, False)
    summarizer.web_visualize()
    network.sess.close()


def compare_real(data_path):
    load_data = pd.read_pickle(data_path)
    s, f = 700, 1000
    xs = load_data.iloc[s:f, 9:]
    ys = load_data.a[s:f]
    network_saver = tfnn.NetworkSaver()
    restore_path = '/tmp/'
    network, input_filter = network_saver.restore(restore_path)
    prediction = network.predict(input_filter.filter(xs))
    plt.plot(np.arange(xs.shape[0]), prediction, 'r--', label='predicted')
    plt.plot(np.arange(xs.shape[0]), ys, 'k-', label='real')
    plt.legend(loc='best')
    plt.show()
    network.sess.close()


class Car:
    def __init__(self, p):
        self.acs = [0]
        self.vs = [0]
        self.ps = [p]
        self.ss = [15]


def test():
    network_saver = tfnn.NetworkSaver()
    restore_path = '/tmp/'
    network, input_filter = network_saver.restore(restore_path)

    cars = []
    for i in range(8):
        cars.append(Car(i*-15))

    for i in range(40*10):
        for j in range(len(cars)):
            if j == 0:
                if i < 1*10:
                    a = 0
                elif 1*10 <= i < 8*10:
                    a = 1.5
                elif 8*10 <= i < 20*10:
                    a = 0
                elif 20*10 <= i < 22*10:
                    a = -3
                elif 22*10 <= i < 25*10:
                    a = -1.5
                else:
                    a = 0
                cars[0].ps.append(cars[0].ps[-1] + cars[0].vs[-1] * 0.1 + 1/2*cars[0].acs[-1]*0.1**2)
                cars[0].vs.append(cars[0].vs[-1]+0.1*a)
                cars[0].acs.append(a)
            else:
                if i <= 1*10:
                    a = 0
                else:
                    ss_data = cars[j].ss[-10:]
                    ss_data.reverse()
                    vs_data = cars[j].vs[-10:]
                    vs_data.reverse()
                    vs_l_data = cars[j-1].vs[-10:]
                    vs_l_data.reverse()
                    xs_data = np.array(ss_data+vs_data+vs_l_data)
                    a = network.predict(input_filter.filter(xs_data))

                    if a < -4:
                        a = 0

                cars[j].ps.append(cars[j].ps[-1] + cars[j].vs[-1] * 0.1 + 1 / 2 * cars[j].acs[-1] * 0.1 ** 2)
                cars[j].vs.append(cars[j].vs[-1] + 0.1 * a)
                cars[j].acs.append(a)
                cars[j].ss.append(cars[j-1].ps[-1]-cars[j].ps[-1])

    xs = list(range(40 * 10 + 1))
    plt.figure(1)
    plt.subplot(411)
    for i in range(len(cars)):
        if i == 0:
            plt.plot(xs, cars[i].ps, 'k-')
        else:
            plt.plot(xs, cars[i].ps, 'r--')
    plt.ylabel('p (m)')
    # plt.legend(loc='best')
    plt.grid()

    plt.subplot(412)
    for i in range(len(cars)):
        if i == 0:
            plt.plot(xs, cars[i].acs, 'k-')
        else:
            plt.plot(xs, cars[i].acs, 'r--')
    plt.ylabel('a (m/s^2)')
    # plt.legend(loc='best')
    plt.grid()
    plt.subplot(413)
    for i in range(len(cars)):
        if i == 0:
            plt.plot(xs, cars[i].vs, 'k-')
        else:
            plt.plot(xs, cars[i].vs, 'r--')
    plt.ylabel('v (m/s)')
    # plt.legend(loc='best')
    plt.grid()
    plt.subplot(414)
    for i in range(1, len(cars)):
        plt.plot(xs, cars[i].ss, 'r--')
    plt.ylabel('space (m)')
    # plt.legend(loc='best')
    plt.grid()
    plt.show()


if __name__ == '__main__':
    path = r'road data/train_I80_lane1.pickle'
    train(path)
    # compare_real(path)
    # test()