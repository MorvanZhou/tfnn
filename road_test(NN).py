import tfnn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def train(data_path, duration, save_to='/tmp/'):
    # load_data = pd.read_pickle(data_path).iloc[:10000, :]
    # xs = load_data.iloc[:, -60:]
    # load_data = pd.read_csv(data_path, index_col=0).dropna()

    load_data = pd.DataFrame()
    for lane_path in data_path:
        lane_data = pd.read_pickle(lane_path).dropna().astype(np.float32)
        load_data = pd.concat([load_data, lane_data], axis=0, ignore_index=True)
    xs = pd.concat([
        # load_data.iloc[:, -60-int(duration*10):-60],        # speed
        load_data.iloc[:, -40-int(duration*10):-40],        # leader speed
        load_data.iloc[:, -20-int(duration*10):-20],        # spacing
        load_data.iloc[:, -int(duration*10):]              # relative speed
        ], axis=1)
    print(xs.shape)
    print(xs.head(2))
    print('sample size:', load_data.shape[0])
    ys = load_data['deri_v']
    # ys = load_data['deri_a_clipped']
    # ys = load_data['delta_x']
    data = tfnn.Data(xs, ys, name='road_data')

    network = tfnn.RegNetwork(xs.shape[1], 1, do_dropout=False)
    n_data = network.normalizer.minmax_fit(data)
    t_data, v_data = n_data.train_test_split(0.8)
    # the number of hidden unit is 2 * xs features
    network.add_hidden_layer(100, activator=tfnn.nn.relu, dropout_layer=False)
    network.add_hidden_layer(10, activator=tfnn.nn.relu, dropout_layer=False)
    network.add_output_layer(activator=None, dropout_layer=False)
    global_step = tfnn.Variable(0, trainable=False)
    # lr = tfnn.train.exponential_decay(0.001, global_step, 5000, 0.9, staircase=False)
    optimizer = tfnn.train.AdamOptimizer(0.001)
    network.set_optimizer(optimizer, global_step)
    evaluator = tfnn.Evaluator(network)
    summarizer = tfnn.Summarizer(network, save_path='/tmp/', include_test=True)

    for i in range(30000):
        b_xs, b_ys = t_data.next_batch(50, loop=True)
        network.run_step(b_xs, b_ys, 0.5)
        if i % 2000 == 0:
            print(evaluator.compute_cost(v_data.xs, v_data.ys))
            summarizer.record_train(b_xs, b_ys, i, 0.5)
            summarizer.record_test(v_data.xs, v_data.ys, i)
            # evaluator.plot_regression_linear_comparison(v_data.xs, v_data.ys, continue_plot=True)
    network.save(path='tmp', name=save_to, replace=True)
    network.sess.close()
    evaluator.hold_plot()
    # summarizer.web_visualize()


def compare_real(path, duration, model_path='tmp', model='/model'):
    # load_data = pd.read_pickle(data_path)
    load_data = pd.read_pickle(path).dropna().astype(np.float32)
    # load_data = pd.read_csv(path, index_col=0).dropna()
    s = 11000
    f = s + 500
    # xs = load_data.iloc[s:f, -60:]
    xs = pd.concat([load_data.iloc[s:f, -60 - int(duration*10):-60],    # speed
                    load_data.iloc[s:f, -40 - int(duration*10):-40],    # leader speed
                    load_data.iloc[s:f, -20 - int(duration*10):-20]],    # spacing
                    # load_data.iloc[s:f, -int(duration * 10):]],           # relative speed
                    axis=1)
    ys = load_data.deri_a_clipped[s:f]
    # ys = load_data.a[s:f]
    network_saver = tfnn.NetworkSaver()
    plt.plot(np.arange(xs.shape[0]), ys, 'k-', label='real')
    network = network_saver.restore(name=model, path=model_path)
    prediction = network.predict(network.normalizer.fit_transform(xs))
    plt.plot(np.arange(xs.shape[0]), prediction, 'r--', label='predicted')
    network.sess.close()
    plt.legend(loc='best')
    plt.show()


class Car:
    def __init__(self, p):
        self.acs = [0]
        self.vs = [0]
        self.ps = [p]
        self.ss = [15]


def test(duration, model_path='tmp', model='/model'):
    network_saver = tfnn.NetworkSaver()
    network = network_saver.restore(name=model, path=model_path)
    test_time = 60
    duration = int(10 * duration)
    cars = []
    for i in range(8):
        cars.append(Car(i*-15))

    for i in range(test_time*10):
        for j in range(len(cars)):
            if j == 0:
                if i < 1*10:
                    a = 0
                elif 1*10 <= i < 6*10:
                    a = 1
                elif 6*10 <= i < 20*10:
                    a = 0.5
                elif 20*10 <= i < 25*10:
                    a = -2
                elif 25*10 <= i < 30*10:
                    a = 0
                elif 30*10 <= i < 35*10:
                    a = 2
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
                if i <= 1*duration:
                    a = 0
                else:
                    ss_data = cars[j].ss[-duration:]
                    vs_data = cars[j].vs[-duration:]
                    vs_l_data = cars[j-1].vs[-duration:]
                    xs_data = np.array(vs_data+vs_l_data+ss_data)
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


def traj_comparison(data_path, duration, id, model_path='tmp', model='/model',
                    on_test=True, predict='a'):
    df = pd.read_pickle(data_path)[['filter_position', 'Vehicle_ID', 'Frame_ID',
                                    'deri_v', 'deri_a_clipped']].dropna().astype(np.float32)
    df = df[df['filter_position'] < 380]
    ids = np.unique(df.Vehicle_ID)
    filter_ids = ids[ids >= id]
    ps = pd.DataFrame()
    vs = pd.DataFrame()
    accs = pd.DataFrame()
    for i, car_id in enumerate(filter_ids):
        if i > 8:
            break
        car_data = df[df['Vehicle_ID'] == car_id]
        car_data = car_data.set_index(['Frame_ID'])
        car_position = car_data['filter_position']
        car_speed = car_data['deri_v']
        car_acc = car_data['deri_a_clipped']
        # shape (time, n_car) for positions
        ps = pd.concat([ps, car_position.rename(i)], axis=1)
        # shape (time, n_car) for speeds
        vs = pd.concat([vs, car_speed.rename(i)], axis=1)
        accs = pd.concat([accs, car_acc.rename(i)], axis=1)

    network_saver = tfnn.NetworkSaver()
    network = network_saver.restore(name=model, path=model_path)
    test_ps = ps.copy()
    test_vs = vs.copy()
    test_accs = accs.copy()

    for i in range(1, 9):
        end_f_id = test_ps.iloc[:, i-1].dropna().index[-1]
        for t in test_ps.iloc[:, i].dropna().index[int(duration*10):]:
            if t == end_f_id:
                break
            # index from test data
            """if use the real ps and vs in here, the predicted acceleration is very close,
            but it has the accumulated error, which will result in a big change in position data.
            If use test_ps, and test_vs, which will depend on the data generated by last time (from prediction).
            the acceleration error will be greater then last method (using real data directly),
            but the position error will be less than last method."""
            if on_test:
                # depend on test
                p_data = test_ps.loc[t - int(duration*10): t-1, i]
                v_data = test_vs.loc[t - int(duration*10): t-1, i]
            else:
                # depend on real
                p_data = ps.loc[t - int(duration * 10): t - 1, i]
                v_data = vs.loc[t - int(duration * 10): t - 1, i]
            # keep index from real data for leader
            pl_data = ps.loc[t-int(duration*10): t-1, i-1]
            vl_data = vs.loc[t-int(duration*10): t-1, i-1]
            dx_data = pl_data - p_data
            dv_data = vl_data - v_data
            # speed, leader_speed, dx, dv
            # [furthest, nearest]
            if predict == 'a':
                # speed, leader_speed, dx
                # [furthest, nearest]
                input_data = pd.concat([v_data, vl_data, dx_data])
                a = network.predict(network.normalizer.fit_transform(input_data))
                new_speed = test_vs.loc[t - 1, i] + a * 0.1
                new_speed = 0 if new_speed < 0 else new_speed
                test_ps.loc[t, i] = test_ps.loc[t - 1, i] + (test_vs.loc[t - 1, i] + new_speed) / 2 * 0.1
                test_vs.loc[t, i] = new_speed
                test_accs.loc[t, i] = a
            elif predict == 'v':
                # leader_speed, dx, dv
                # [furthest, nearest]
                input_data = pd.concat([
                    # v_data,
                    vl_data,
                    dx_data,
                    dv_data,
                ])
                new_speed = network.predict(network.normalizer.fit_transform(input_data))
                new_speed = 0 if new_speed < 0 else new_speed
                test_ps.loc[t, i] = test_ps.loc[t - 1, i] + (test_vs.loc[t - 1, i] + new_speed) / 2 * 0.1
                test_vs.loc[t, i] = new_speed
                test_accs.loc[t, i] = (new_speed - test_vs.loc[t - 1, i]) / .1
            elif predict == 'delta_x':
                """
                !!!!
                bad idea
                !!!!
                """
                # speed, leader_speed, dx
                # [furthest, nearest]
                input_data = pd.concat([v_data, vl_data, dx_data])
                delta_x = network.predict(network.normalizer.fit_transform(input_data))
                delta_x = 0 if delta_x < 0 else delta_x
                # delta_x = (v0+v1)*t/2
                # so v1 = delta_x*2/t - v0
                test_ps.loc[t, i] = test_ps.loc[t-1, i] + delta_x
                test_vs.loc[t, i] = delta_x/0.1
                test_accs.loc[t, i] = (test_vs.loc[t, i] - test_vs.loc[t-1, i])/0.1

    plt.figure(0)
    plt.title('Position')
    plt.plot(ps, 'k-')
    plt.plot(test_ps.iloc[:, 1:], 'r--')
    plt.ylim((0, 400))

    f, ax = plt.subplots(8, 1)
    f.suptitle('Velocity')
    for i in range(8):
        ax[i].plot(vs.iloc[:, i + 1], 'k-')
        ax[i].plot(test_vs.iloc[:, i + 1], 'r--')

    f, ax = plt.subplots(8, 1)
    f.suptitle('Acceleration')
    for i in range(8):
        ax[i].plot(accs.iloc[:, i+1], 'k-')
        ax[i].plot(test_accs.iloc[:, i+1], 'r--')

    f, ax = plt.subplots(8, 1)
    f.suptitle('test real acceleration diff cumsum')
    for i in range(8):
        ax[i].plot((test_accs.iloc[:, i + 1]-accs.iloc[:, i + 1]).cumsum(), 'k-')

    plt.show()


def oscillation(path, duration, id,  model_path='tmp', model='/model'):
    df = pd.read_pickle(path)[['filter_position', 'Vehicle_ID',
                               'Frame_ID', 'deri_v']].dropna().astype(np.float32)
    all_speed = df['deri_v'][(df['filter_position']>=100) & (df['filter_position']<=300)]
    speed_mean = all_speed.mean()
    speed_std = all_speed.std()
    df = df[df['filter_position'] >= 50]
    # df = df[df['Frame_ID'] < 3000]
    leader = df[df['Vehicle_ID'] == id]
    ids = np.unique(df.Vehicle_ID)
    filter_ids = ids[ids >= id]
    positions = pd.DataFrame(index=range(1800, 2800))
    for i, car_id in enumerate(filter_ids):
        if i > 10:
            break
        car_data = df[df['Vehicle_ID'] == car_id]
        car_data = car_data.set_index(['Frame_ID'])
        car_position = car_data['filter_position']
        positions = pd.concat([positions, car_position], axis=1)

    # plt.plot(positions)
    # plt.show()

    network_saver = tfnn.NetworkSaver()
    network = network_saver.restore(name=model, path=model_path)

    all_dx = positions.diff(axis=1).stack()
    dx_mean = all_dx.mean()
    dx_std = all_dx.std()
    all_speeds = pd.DataFrame()
    all_speeds = pd.concat([all_speeds, leader['deri_v'].dropna().rename(0)], axis=1)
    leader_speed_append = all_speeds.iloc[-1, 0] * pd.Series(np.ones(400))
    all_speeds = pd.concat([all_speeds, leader_speed_append], axis=0).reset_index(drop=True)

    all_ps = pd.DataFrame()
    all_ps = pd.concat([all_ps, leader['filter_position'].dropna().rename(0)], axis=1)
    position_change = all_speeds.iloc[-1, 0] * 0.1
    leader_position_append = all_ps.iloc[-1, 0] + position_change * pd.Series(np.arange(1, 401))
    all_ps = pd.concat([all_ps, leader_position_append], axis=0).reset_index(drop=True)
    for i in range(1, 10):
        distance_noise = (np.random.normal(dx_mean, dx_std))
        distance_noise = np.min([-10, distance_noise])
        distance_noise = np.max([-30, distance_noise])
        individual_p = all_ps.iloc[:int(duration*10), i-1] + distance_noise
        all_ps = pd.concat([all_ps, individual_p.rename(i)], axis=1)
        speed_noise = (np.random.normal(speed_mean, speed_std))
        speed_noise = np.max([5, speed_noise])
        speed_noise = np.min([20, speed_noise])
        individual_speed = pd.DataFrame(np.zeros((int(duration*10), 1)) + speed_noise,
                                        columns=[i])
        all_speeds = pd.concat([all_speeds, individual_speed], axis=1, ignore_index=True)

        for t in range(int(duration*10), all_ps.shape[0]-1):
            # speed, leader speed, dx, dv
            # [furthest, nearest]
            v = all_speeds.iloc[t-int(duration*10): t, i]
            v_l = all_speeds.iloc[t-int(duration*10): t, i-1]
            dx = all_ps.iloc[t-int(duration*10): t, i-1] - all_ps.iloc[t-int(duration*10): t, i]
            # dv = v_l - v
            input_data = pd.concat([v, v_l, dx])
            a = network.predict(network.normalizer.fit_transform(input_data))
            new_speed = all_speeds.iloc[t-1, i] + a * 0.1
            new_speed = 0 if new_speed < 0 else new_speed
            all_ps.iloc[t, i] = all_ps.iloc[t-1, i] + (all_speeds.iloc[t-1, i]+new_speed)/2*0.1
            all_speeds.iloc[t, i] = new_speed

    plt.plot(all_ps,)
    plt.ylim((50, 500))
    plt.xlim((0, 800))
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
    # tfnn.set_random_seed(11)
    # np.random.seed(11)
    which_lane = [1, 3, 4]
    path = ['datasets/I80-0400_lane%i.pickle' % lane for lane in which_lane]
    duration = 1
    train(path, duration, save_to='/model%i/' % int(duration*10))

    # test_path = 'datasets/I80-0400_lane2.pickle'
    # compare_real(test_path, duration, model_path='tmp', model='/model%i/' % int(duration*10))

    # test(duration,  model_path='tmp', model='/model%i/' % int(duration*10))

    # cross_validation(path)

    # 512, 517, 418, 121, 191, 242, 392
    lane_path = 'datasets/I80-0400_lane2.pickle'
    traj_comparison(lane_path, duration, id=890, model='/model%i/' % int(duration*10),
                    on_test=True, predict='v')
    # oscillation(lane_path, duration, id=890, model='/model%i/' % int(duration*10))