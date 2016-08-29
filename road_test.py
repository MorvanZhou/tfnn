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
        lane_data = pd.read_pickle(lane_path).dropna()
        load_data = pd.concat([load_data, lane_data], axis=0, ignore_index=True)
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
    t_data, v_data = n_data.train_test_split(0.9)
    # the number of hidden unit is 2 * xs features
    network.add_hidden_layer(100, activator=tfnn.nn.tanh, dropout_layer=True)
    network.add_hidden_layer(100, activator=tfnn.nn.tanh, dropout_layer=True)
    network.add_hidden_layer(10, activator=tfnn.nn.tanh, dropout_layer=True)
    network.add_output_layer(activator=None, dropout_layer=False)
    global_step = tfnn.Variable(0, trainable=False)
    lr = tfnn.train.exponential_decay(0.001, global_step, 2000, 0.9, staircase=False)
    optimizer = tfnn.train.AdamOptimizer(lr)
    network.set_optimizer(optimizer, global_step)
    evaluator = tfnn.Evaluator(network)
    summarizer = tfnn.Summarizer(network, save_path='/tmp/', include_test=True)

    for i in range(40000):
        b_xs, b_ys = t_data.next_batch(50, loop=True)
        network.run_step(b_xs, b_ys, 0.5)
        if i % 1000 == 0:
            print(evaluator.compute_cost(v_data.xs, v_data.ys))
            summarizer.record_train(b_xs, b_ys, i, 0.5)
            summarizer.record_test(v_data.xs, v_data.ys, i)
            evaluator.regression_plot_linear_comparison(v_data.xs, v_data.ys, continue_plot=True)
    network.save(path='tmp', name=save_to)
    network.sess.close()
    summarizer.web_visualize()


def compare_real(path, duration, model_path='tmp', model='/model'):
    # load_data = pd.read_pickle(data_path)
    load_data = pd.read_pickle(path).dropna().iloc[-10000:, :]
    # load_data = pd.read_csv(path, index_col=0).dropna()
    s = 8000
    f = s + 1000
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

    # for i in ['02','04', '06', '08', '10', '15']:
    #     duration = int(i)/10
    #     xs = pd.concat([load_data.iloc[s:f, -60 - int(duration * 10):-60],  # speed
    #                     load_data.iloc[s:f, -40 - int(duration * 10):-40],  # leader speed
    #                     load_data.iloc[s:f, -20 - int(duration * 10):-20]],  # spacing
    #                    # load_data.iloc[s:f, -int(duration * 10):]],           # relative speed
    #                    axis=1)
    #     model = '/model'+i+'/'
    #     network = network_saver.restore(model)
    #     prediction = network.predict(network.normalizer.fit_transform(xs))
    #     plt.plot(np.arange(xs.shape[0]), prediction, '--', label=model)
    #     network.sess.close()

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


def traj_comp(path, duration, id,  model_path='tmp', model='/model'):
    df = pd.read_pickle(path)

    vehicle = df[df.Vehicle_ID == id]
    plt.plot(vehicle.Frame_ID, vehicle.filter_position, c='r', label=id)
    p_vehicle = vehicle
    for _ in range(40):
        p_id = p_vehicle.Vehicle_ID.iloc[0]
        followers = (df[df.Preceding_Vehicle == p_id]).Vehicle_ID.unique()
        for follower_id in followers:
            follower = df[df.Vehicle_ID == follower_id]
            plt.plot(follower.Frame_ID, follower.filter_position, label=follower_id)
        p_vehicle = df[df.Vehicle_ID == followers[0]]
    plt.legend()
    plt.show()


    # car_517 = df[df.Vehicle_ID == 517].loc[:, ['Frame_ID', 'filter_position', 'deri_v']].dropna()
    # car_514 = df[df.Vehicle_ID == 514].loc[:, ['Frame_ID', 'filter_position', 'deri_v']].dropna()
    # car_522 = df[df.Vehicle_ID == 522].loc[:, ['Frame_ID', 'filter_position', 'deri_v']].dropna()
    # car_525 = df[df.Vehicle_ID == 525].loc[:, ['Frame_ID', 'filter_position', 'deri_v']].dropna()
    # start_point = 1911
    # # combine car 517 and 514 as the leader for machine learning
    # leader = pd.concat([car_517[car_517.Frame_ID < 2102], car_514], axis=0)
    # leader = leader[leader.Frame_ID >= start_point]
    #
    # # filter car 522 to the same start point as the leader
    # car_522 = car_522[car_522.Frame_ID >= start_point]
    # car_525 = car_525[car_525.Frame_ID >= start_point]
    #
    # plt.plot(car_517.Frame_ID, car_517.filter_position, label=517)
    # plt.plot(car_514.Frame_ID, car_514.filter_position, label=514)
    # # plt.plot(leader.Frame_ID, leader.filter_position, label='leader')
    # plt.plot(car_522.Frame_ID, car_522.filter_position, label=522)
    # plt.plot(car_525.Frame_ID, car_525.filter_position, label=525)
    #
    #
    # network_saver = tfnn.NetworkSaver()
    # network = network_saver.restore(name=model, path=model_path)
    #
    # tested_vehicle1 = car_522.iloc[:int(duration*10), :]
    # position1 = car_522.filter_position.iloc[int(duration*10)-1]
    # speed1 = car_522.deri_v.iloc[int(duration*10)-1]
    #
    # tested_vehicle2 = car_525.iloc[:int(duration * 10), :]
    # position2 = car_525.filter_position.iloc[int(duration * 10) - 1]
    # speed2 = car_525.deri_v.iloc[int(duration * 10) - 1]
    #
    # tested_vehicles = [tested_vehicle1, tested_vehicle2]
    # positions = [position1, position2]
    # speeds = [speed1, speed2]
    # leaders = [leader, tested_vehicle1]
    #
    # for index in range(int(duration*10), len(leader)):
    #     for t_car in range(2):
    #         if t_car == 1:
    #             leaders[1] = tested_vehicles[0]
    #         v_data = tested_vehicles[t_car].deri_v.iloc[index-int(duration*10): index]
    #         v_l_data = leaders[t_car].deri_v.iloc[index-int(duration*10): index]
    #         dx_data = pd.Series(leaders[t_car].filter_position.iloc[index-int(duration*10): index].values \
    #                   - tested_vehicles[t_car].filter_position.iloc[index-int(duration*10): index].values)
    #         xs_data = v_data.append([v_l_data, dx_data]).reset_index(drop=True)
    #         a = network.predict(network.normalizer.fit_transform(xs_data))
    #         positions[t_car] += 1/2*a*0.1**2 + v_data.iloc[-1]*0.1
    #         speeds[t_car] += a*0.1
    #         f_id = leader.Frame_ID.iloc[index]
    #         tested_vehicles[t_car] = pd.concat([tested_vehicles[t_car],
    #                                     pd.DataFrame([[f_id, positions[t_car], speeds[t_car]]],
    #                                                  columns=tested_vehicle1.columns)],
    #                                    axis=0,
    #                                    ignore_index=True)
    #
    # for t_car, c_id in zip(range(2), [522, 525]):
    #     plt.plot(tested_vehicles[t_car].Frame_ID, tested_vehicles[t_car].filter_position, 'k--', label='test%s' % c_id)
    # plt.legend(loc='best')
    # plt.show()
    # # print(follower.loc[:, ['Vehicle_ID', 'Frame_ID', 'filter_position']])

def oscillation(path, duration, id,  model_path='tmp', model='/model'):
    df = pd.read_pickle(path)[['filter_position', 'Vehicle_ID', 'Frame_ID', 'deri_v']].dropna()
    df = df[df['filter_position'] >= 280]
    df = df[df['Frame_ID'] < 3000]
    leader = df[df['Vehicle_ID'] == id]
    ids = np.unique(df.Vehicle_ID)
    filter_ids = ids[ids >= id]
    positions = pd.DataFrame(index=range(1400, 2200))
    speeds = pd.DataFrame(index=range(1400, 2200))
    for i, car_id in enumerate(filter_ids):
        if i > 20:
            break
        car_data = df[df['Vehicle_ID'] == car_id]
        car_data = car_data.set_index(['Frame_ID'])
        car_position = car_data['filter_position']
        car_speed = car_data['deri_v']
        positions = pd.concat([positions, car_position], axis=1)
        speeds = pd.concat([speeds, car_speed.rename(i)], axis=1)

    plt.plot(positions)
    plt.show()

    positions.columns = range(21)
    end_f_ids = positions.idxmax(axis=0)
    positions = positions[positions < 350]
    positions = positions[positions.max().dropna().index]
    end_f_ids = end_f_ids[positions.max().dropna().index]
    positions.columns = range(19)
    end_f_ids.index = range(19)
    start_f_ids = positions.idxmin(axis=0)
    start_ps = pd.DataFrame()
    start_vs = pd.DataFrame()

    for i, f_id in enumerate(start_f_ids):
        if i == 0:
            start_p = df.loc[:, ['filter_position', 'Frame_ID']][df['Vehicle_ID'] == id].set_index('Frame_ID')
            start_v = df.loc[:, ['deri_v', 'Frame_ID']][df['Vehicle_ID'] == id].set_index('Frame_ID')
        else:
            start_p = positions.loc[f_id:f_id+int(duration*10), i]
            start_v = speeds.loc[f_id: f_id + int(duration * 10), i]
        start_ps = pd.concat([start_ps, start_p], axis=1, ignore_index=True)
        start_vs = pd.concat([start_vs, start_v], axis=1, ignore_index=True)

    network_saver = tfnn.NetworkSaver()
    network = network_saver.restore(name=model, path=model_path)

    for i in range(1, 19):
        for t in range(int(start_f_ids.iloc[i]+int(duration*10)), int(end_f_ids.iloc[i])):
            speed = start_vs.loc[t-int(duration*10)+1: t, i]
            leader_speed = start_vs.loc[t-int(duration*10)+1: t, i-1]
            p = start_ps.loc[t-int(duration*10)+1: t, i]
            leader_p = start_ps.loc[t-int(duration*10)+1: t, i-1]
            dx = leader_p - p
            # speed, leader speed, dx
            # [furthest, nearest]
            input_data = pd.concat([speed, leader_speed, dx])
            if len(input_data) > 30:
                pass
            a = network.predict(network.normalizer.fit_transform(input_data))
            start_ps.loc[t+1, i] = start_ps.loc[t, i] + speed.iloc[-1]*0.1 + 1/2*a*0.1**2
            start_vs.loc[t+1, i] = start_vs.loc[t, i] + a*0.1
        plt.plot(start_ps)
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
    tfnn.set_random_seed(100)
    np.random.seed(101)
    which_lane = [1, 2, 3]
    path = ['datasets/I80-0400_lane%i.pickle' % lane for lane in which_lane]
    # path = 's3.pickle'
    duration = 1
    # train(path, duration, save_to='/model%i/' % int(duration*10))

    test_path = 'datasets/I80-0400_lane2.pickle'
    # compare_real(test_path, duration, model_path='tmp', model='/model%i/' % int(duration*10))

    # test(duration,  model_path='tmp', model='/model%i/' % int(duration*10))

    # cross_validation(path)

    # 512, 517, 418, 121, 191, 242, 392
    lane4_path = 'datasets/I80-0400_lane4.pickle'
    # traj_comp(lane4_path, duration, id=418,  model_path='tmp', model='/model%i/' % int(duration*10))
    # df = pd.read_pickle(lane4_path)
    # ids = np.unique(df.Vehicle_ID)
    # for id in ids:
    #     print(id)
    #     traj_comp(lane4_path, duration, id, model_path='tmp', model='/model%i/' % int(duration*10))
    oscillation(lane4_path, duration, id=418, model='model10', model_path='tmp')