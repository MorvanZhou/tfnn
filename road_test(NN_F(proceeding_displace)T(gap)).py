import tfnn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


################################################
class Dataset(object):
    def __init__(self, path):
        df = pd.read_pickle(path)
        df = df[(df['Lane_Identification'] >= 2) & (df['Lane_Identification'] <= 6)]
        self._road_data = df.loc[:,
                          ['Vehicle_ID', 'deri_v', 'displacement', 'dx', 'dv',
                           'proceeding_displacement','deri_a_clipped', 'v_l']].dropna()
        self._car_ids = np.unique(self._road_data['Vehicle_ID'])
        self._id_index = None

    def next(self, batch_size, time_steps, predict):
        all_f, all_t = [], []
        for _ in range(batch_size):
            print(_)
            b_features, b_targets = self._get_sample(time_steps, predict)
            all_f.append(b_features)
            all_t.append(b_targets)
        b_features = np.vstack(all_f)
        b_targets = np.vstack(all_t)
        print('done')
        return [b_features, b_targets]

    def _get_sample(self, time_steps, predict):
        while True:
            id_index = np.random.choice(np.arange(len(self._car_ids)))
            car_id = self._car_ids[id_index]
            car_data = self._road_data[self._road_data['Vehicle_ID'] == car_id]
            if car_data.shape[0] >= time_steps+1:
                start_time = np.random.choice(range(car_data.shape[0])[: - time_steps - 1])
                end_time = start_time + time_steps
                if predict == 'displacement':
                    features = car_data['proceeding_displacement'].iloc[start_time: end_time] / 3.5 - 0.5,  # displacement
                    targets = np.array([car_data['gap'].iloc[end_time + 1]])
                else:
                    raise ValueError('not support')
                break
        return [features, targets]


def train_(train_config, predict, test_config):
    # for instantly data generation, use next 2 lines:
    # data = Dataset(train_config.data_path)
    # v_xs, v_ys = data.next(2000, train_config.time_steps, predict)
    all_data = pd.read_pickle(train_config.data_path)
    X = all_data[list(range(20-train_config.time_steps, 20))]
    normal_data = tfnn.Data(X, all_data.iloc[:, -1])
    t_data, v_data = normal_data.train_test_split(0.95)
    train_nn = tfnn.RegNetwork(t_data.xs.shape[1], t_data.ys.shape[1], do_dropout=False)
    h1 = tfnn.HiddenLayer(train_config.hidden_layers[0], activator='relu')
    out = tfnn.OutputLayer(activator='relu')
    train_nn.build_layers([h1, out])
    train_nn.set_optimizer('adam')
    train_nn.set_learning_rate(train_config.learning_rate,
                               exp_decay=dict(decay_steps=train_config.decay_steps,
                                              decay_rate=train_config.decay_rate))
    evaluator = tfnn.Evaluator(train_nn)
    evaluator.set_scale_monitor(['learning rate', 'cost'], figsize=(7, 5))
    evaluator.set_data_fitting_monitor()

    for step in range(ITER_STEPS):
        b_xs, b_ys = t_data.next_batch(train_config.batch_size)
        train_nn.run_step(b_xs, b_ys)
        if step % PLOT_LOOP == 0:
            evaluator.monitoring(b_xs, b_ys, v_xs=v_data.xs, v_ys=v_data.ys)
        if step % TEST_LOOP == 0:
            test_(test_config, 890, PREDICT, model=train_nn)
    print('finish training')
    evaluator.hold_plot()
    train_nn.save('nn', 'tmp', replace=True)
    return train_nn


def test_(test_config, id, predict, model=None):
    if model is None:
        none_model = True
        saver = tfnn.NetworkSaver()
        model = saver.restore(name=test_config.restore_model, path=test_config.restore_path)
    else:
        none_model = False

    ps, vs, accs, proceeding_dispms = test_config.data
    test_ps = ps.copy()
    test_vs = vs.copy()

    for i in range(1, test_config.sim_car_num):
        end_f_id = ps.iloc[:, i - 1].dropna().index[-test_config.time_steps]  # for preceding vehicle
        has_been_dropped = False
        for t in ps.iloc[:, i].dropna().index:  # for current vehicle
            if t == end_f_id:
                # filter out the original value
                test_ps.loc[t + 1:, i] = None
                test_vs.loc[t + 1:, i] = None
                break
            proceeding_dispms_data = proceeding_dispms.loc[t: t + test_config.time_steps - 1, i]

            if predict == 'gap':
                input_data = proceeding_dispms_data * test_config.displacement_scale + test_config.displacement_bias,  # displacement
                new_gap = model.predict(input_data)
                test_ps.loc[t + test_config.time_steps, i] = ps.loc[t+test_config.time_steps, i-1] - new_gap
                if not has_been_dropped:
                    test_ps.loc[:t + test_config.time_steps-1, i] = None
                    test_vs.loc[:t + test_config.time_steps-1, i] = None
                    has_been_dropped = True
            else:
                raise ValueError('not support')

    plt.figure(0)
    if not none_model:
        plt.ion()
        plt.cla()
    plt.title('Position')
    plt.plot(ps, 'k-')
    plt.plot(test_ps.iloc[:, 1:], 'r--')
    plt.ylim((0, 400))
    plt.show()

def set_seed(seed):
    tfnn.set_random_seed(11)
    np.random.seed(11)


class TrainConfig(object):
    # 0400-0415: much oscillation occurs on lane 2,4,5,6
    # data_path = 'datasets/I80-0400-0415-filter_0.8_gap_displacement.pickle'

    # 0500-0515: much oscillation occurs on lane 2,3,4,5,6
    # data_path = 'datasets/I80-0500-0515-filter_0.8_T_v_ldxdvhdisplace.pickle'

    # random generated 30000 data 2 seconds duration
    data_path = 'datasets/I80-0500-0515-filter_0.8_T_F(proceeding_displace)T(gap)_4NN_300000.pickle'

    batch_size = 50
    time_steps = 5
    displacement_scale = 1/3.5
    displacement_bias = - 0.5
    gap_scale = 1/86
    gap_bias = -0.5
    hidden_layers = [50]
    is_training = True
    keep_prob = 1
    learning_rate = 1e-3
    decay_steps = 2000
    decay_rate = 1


class TestConfig(object):
    data_path = 'datasets/I80-0400_lane2_proceeding_position&displacement.pickle'
    batch_size = 1
    time_steps = 5
    displacement_scale = 1 / 3.5
    displacement_bias = - 0.5
    gap_scale = 1 / 86
    gap_bias = -0.5
    output_size = 1
    is_training = False
    restore_path = 'tmp'
    restore_model = 'nn'
    sim_car_num = 9
    start_car_id = 890

    def __init__(self):
        self.data = self.get_test_date()

    def get_test_date(self):
        df = pd.read_pickle(self.data_path)[['filter_position', 'Vehicle_ID', 'Frame_ID', 'displacement',
                                             'proceeding_displacement', 'deri_v', 'deri_a_clipped']].dropna().astype(np.float32)
        df = df[df['filter_position'] < 380]
        ids = np.unique(df.Vehicle_ID)
        filter_ids = ids[ids >= self.start_car_id]
        ps = pd.DataFrame()
        vs = pd.DataFrame()
        accs = pd.DataFrame()
        proceeding_dispms = pd.DataFrame()
        for i, car_id in enumerate(filter_ids):
            if i >= self.sim_car_num:
                break
            car_data = df[df['Vehicle_ID'] == car_id]
            car_data = car_data.set_index(['Frame_ID'])
            car_position = car_data['filter_position']
            car_speed = car_data['deri_v']
            car_acc = car_data['deri_a_clipped']
            proceeding_car_dispms = car_data['displacement']
            # shape (time, n_car) for positions
            ps = pd.concat([ps, car_position.rename(i)], axis=1)
            # shape (time, n_car) for speeds
            vs = pd.concat([vs, car_speed.rename(i)], axis=1)
            accs = pd.concat([accs, car_acc.rename(i)], axis=1)
            proceeding_dispms = pd.concat([proceeding_dispms, proceeding_car_dispms.rename(i)], axis=1)
        return [ps, vs, accs, proceeding_dispms]

if __name__ == '__main__':
    """
        For this model
        Input: leader displacement;
        Output: gap.
    """
    # set_seed(1)

    ITER_STEPS = 80001
    PLOT_LOOP = 500
    TEST_LOOP = 5000
    PREDICT = 'gap'
    train_config = TrainConfig()
    test_config = TestConfig()

    # # generate training data for NN
    # data = Dataset(train_config.data_path)
    # v_xs, v_ys = data.next(300000, train_config.time_steps, PREDICT)
    # all_data = np.hstack((v_xs, v_ys))
    # pd.DataFrame(all_data).to_pickle('datasets/I80-0500-0515-filter_0.8_T_F-displace_gap_T-displace_4NN_normalized300000.pickle')

    train_nn = train_(train_config, PREDICT, test_config)
    # test_(test_config, 890, PREDICT, on_test=False, model=None)
