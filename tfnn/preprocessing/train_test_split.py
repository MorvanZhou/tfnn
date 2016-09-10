import tfnn
from tfnn.preprocessing.shuffle import shuffle


def train_test_split(data, train_rate=0.7, randomly=True):
    data_copy = data.copy()
    _n_train_samples = int(data.n_samples * train_rate)
    if randomly:
        data_copy = shuffle(data_copy)
    train_data_data = data_copy.data[:_n_train_samples, :]
    test_data_data = data_copy.data[_n_train_samples:, :]
    train_data_data_xs = train_data_data[:, :data.n_xfeatures]
    train_data_data_ys = train_data_data[:, data.n_xfeatures:]
    test_data_data_xs = test_data_data[:, :data.n_xfeatures]
    test_data_data_ys = test_data_data[:, data.n_xfeatures:]
    t_data = tfnn.Data(train_data_data_xs, train_data_data_ys, name='train')
    v_data = tfnn.Data(test_data_data_xs, test_data_data_ys, name='validate')
    return [t_data, v_data]