import tfnn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def train(data_path):
    # load_data = pd.read_pickle(data_path).iloc[:10000, :]
    # xs = load_data.iloc[:, -60:]
    load_data = pd.read_pickle(data_path).dropna()
    # load_data = pd.read_csv(data_path, index_col=0).dropna()
    duration = 0.8    # second
    xs = pd.concat([load_data.iloc[:, -60-int(duration*10):-60],        # speed
                    load_data.iloc[:, -40-int(duration*10):-40],        # leader speed
                    load_data.iloc[:, -20-int(duration*10):-20]],        # spacing
                    # load_data.iloc[:, -int(duration*10):]],             # relative speed
                   axis=1)
    print(xs.shape)
    print(xs.head(2))
    print('sample size:', load_data.shape[0])
    # ys = load_data.a
    ys = load_data.deri_a_clipped
    data = tfnn.Data(xs, ys, name='road_data')

    network = tfnn.RegNetwork(xs.shape[1], 1, do_dropout=False)
    n_data = network.normalizer.minmax_fit(data)
    t_data, v_data = n_data.train_test_split(0.7)
    # the number of hidden unit is 2 * xs features
    network.add_hidden_layer(xs.shape[1]*2, activator=tfnn.nn.relu, dropout_layer=True)
    # network.add_hidden_layer(100, activator=tfnn.nn.relu, dropout_layer=True)
    network.add_output_layer(activator=None, dropout_layer=False)
    global_step = tfnn.Variable(0, trainable=False)
    # lr = tfnn.train.exponential_decay(0.001, global_step, 2000, 0.9)
    optimizer = tfnn.train.AdamOptimizer(0.001)
    network.set_optimizer(optimizer, global_step)
    evaluator = tfnn.Evaluator(network)
    summarizer = tfnn.Summarizer(network, save_path='/tmp/log')

    for i in range(60000):
        b_xs, b_ys = t_data.next_batch(100, loop=True)
        network.run_step(b_xs, b_ys, 0.5)
        if i % 1000 == 0:
            print(evaluator.compute_cost(v_data.xs, v_data.ys))
            summarizer.record_train(b_xs, b_ys, i, 0.5)
            summarizer.record_validate(v_data.xs, v_data.ys, i)
            evaluator.regression_plot_linear_comparison(v_data.xs, v_data.ys, continue_plot=True)
    network.save()
    network.sess.close()
    summarizer.web_visualize()


def compare_real(path):
    # load_data = pd.read_pickle(data_path)
    load_data = pd.read_pickle(path).dropna().iloc[-10000:, :]
    # load_data = pd.read_csv(path, index_col=0).dropna()
    s = 7900
    f = s + 300
    # xs = load_data.iloc[s:f, -60:]
    duration = 0.2
    xs = pd.concat([load_data.iloc[s:f, -60 - int(duration*10):-60],    # speed
                    load_data.iloc[s:f, -40 - int(duration*10):-40],    # leader speed
                    load_data.iloc[s:f, -20 - int(duration*10):-20]],    # spacing
                    # load_data.iloc[s:f, -int(duration * 10):]],           # relative speed
                    axis=1)
    ys = load_data.deri_a_clipped[s:f]
    # ys = load_data.a[s:f]
    network_saver = tfnn.NetworkSaver()
    restore_path = '/tmp/'
    network = network_saver.restore(restore_path)
    prediction = network.predict(network.normalizer.fit_transform(xs))
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
    network = network_saver.restore(restore_path)
    test_time = 60
    cars = []
    for i in range(8):
        cars.append(Car(i*-15))

    for i in range(test_time*10):
        for j in range(len(cars)):
            if j == 0:
                if i < 1*10:
                    a = 0
                elif 1*10 <= i < 15*10:
                    a = 2
                elif 15*10 <= i < 20*10:
                    a = 0
                elif 20*10 <= i < 28*10:
                    a = -3
                elif 28*10 <= i < 30*10:
                    a = 0
                elif 30*10 <= i < 35*10:
                    a = 3
                elif 35 * 10 <= i < 37 * 10:
                    a = 0
                elif 37 * 10 <= i < 45 * 10:
                    a = -1
                elif 45 * 10 <= i < 50 * 10:
                    a = 2
                else:
                    a = 0
                cars[0].ps.append(cars[0].ps[-1] + cars[0].vs[-1] * 0.1 + 1/2*cars[0].acs[-1]*0.1**2)
                v = cars[0].vs[-1] + 0.1 * a
                if v < 0:
                    v = 0
                cars[0].vs.append(v)
                cars[0].acs.append(a)
            else:
                if i <= 1*10:
                    a = 0
                else:
                    ss_data = cars[j].ss[-10:]
                    vs_data = cars[j].vs[-10:]
                    vs_l_data = cars[j-1].vs[-10:]
                    xs_data = np.array(ss_data+vs_data+vs_l_data)
                    a = network.predict(network.normalizer.fit_transform(xs_data))

                cars[j].ps.append(cars[j].ps[-1] + cars[j].vs[-1] * 0.1 + 1 / 2 * cars[j].acs[-1] * 0.1 ** 2)
                v = cars[j].vs[-1] + 0.1 * a
                if v < 0:
                    v = 0
                cars[j].vs.append(v)
                cars[j].acs.append(a)
                cars[j].ss.append(cars[j-1].ps[-1]-cars[j].ps[-1])

    xs = list(range(test_time * 10 + 1))
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


def cross_validation(path):
    seconds_range = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 1.5, 2.0]
    # learning_rates = .1 ** np.arange(1, 6, dtype=np.float32)
    cross_duration_cost = pd.DataFrame()
    cross_duration_r2 = pd.DataFrame()
    for duration in seconds_range:
        tfnn.set_random_seed(111)
        np.random.seed(111)
        load_data = pd.read_pickle(path).dropna()

        # data include v, v_leader, dx
        xs = pd.concat([load_data.iloc[:, -60 - int(duration*10):-60],    # speed
                        load_data.iloc[:, -40 - int(duration*10):-40],    # leader speed
                        load_data.iloc[:, -20 - int(duration*10):-20]],    # spacing
                        # load_data.iloc[:, -int(duration * 10):]],           # relative speed
                        axis=1)
        ys = load_data.deri_a_clipped
        data = tfnn.Data(xs, ys, name='road_data')

        network = tfnn.RegNetwork(xs.shape[1], 1, do_dropout=False)
        n_data = network.normalizer.minmax_fit(data)
        t_data, v_data = n_data.train_test_split(0.7)

        # the number of hidden unit is 2 * xs features
        network.add_hidden_layer(xs.shape[1] * 2, activator=tfnn.nn.relu, dropout_layer=True)
        network.add_output_layer(activator=None, dropout_layer=False)
        global_step = tfnn.Variable(0, trainable=False)
        # done a cross validation about learning rate, the best lr is 0.001
        optimizer = tfnn.train.AdamOptimizer(0.001)
        network.set_optimizer(optimizer, global_step)
        evaluator = tfnn.Evaluator(network)

        # duration_cost = pd.Series(name='%s' % lr)
        # duration_r2 = pd.Series(name='%s' % lr)

        duration_cost = pd.Series(name='%s s' % duration)
        duration_r2 = pd.Series(name='%s s' % duration)

        # duration_cost = pd.Series(name='Test')  #
        # duration_r2 = pd.Series(name='Test')  #
        # train_cost = pd.Series(name='Train')  #
        # train_r2 = pd.Series(name='Train')  #
        for i in range(60000):
            b_xs, b_ys = t_data.next_batch(100, loop=True)
            network.run_step(b_xs, b_ys, 0.5)
            if i % 200 == 0:
                cost = evaluator.compute_cost(v_data.xs, v_data.ys)
                r2 = evaluator.compute_r2_score(v_data.xs, v_data.ys)

                # cost_train = evaluator.compute_cost(t_data.xs, t_data.ys)  #
                # r2_train = evaluator.compute_r2_score(t_data.xs, t_data.ys)  #
                duration_cost.set_value(i, cost)
                duration_r2.set_value(i, r2)

                # train_cost.set_value(i, cost_train) #
                # train_r2.set_value(i, r2_train) #
        cross_duration_cost[duration_cost.name] = duration_cost
        cross_duration_r2[duration_r2.name] = duration_r2

        # cross_duration_cost[train_cost.name] = train_cost   #
        # cross_duration_r2[train_r2.name] = train_r2   #
        network.sess.close()
    cross_duration_cost.plot()
    plt.ylabel('Cost')
    plt.xlabel('Epoch')

    cross_duration_r2.plot()
    plt.ylabel('R2 score')
    plt.xlabel('Epoch')

    final_cost = cross_duration_cost.iloc[-1, :]
    final_r2 = cross_duration_r2.iloc[-1, :]
    plt.figure(3)
    final_cost.plot()
    plt.ylabel('Cost')
    plt.xlabel('Duration')
    plt.figure(4)
    final_r2.plot()
    plt.ylabel('R2 score')
    plt.xlabel('Duration')

    plt.show()

if __name__ == '__main__':
    path = r'I80_lane4.pickle'
    # path = r'/Users/MorvanZhou/Documents/python/2016_05_21_tfnn/road data/train_I80_lane_1_1s.pickle'
    # train(path)
    compare_real(path)
    # test()
    # cross_validation(path)
