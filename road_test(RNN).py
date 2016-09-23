import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

TIME_STEPS = 20
INPUT_SIZE = 3
OUTPUT_SIZE = 1
CELL_SIZE = 10
BATCH_SIZE = 1


road_data = pd.read_pickle('datasets/I80-0400_lane2_4RNN.pickle').dropna()
car_ids = np.unique(road_data['Vehicle_ID'])


def data_gen():
    for id in car_ids:
        car_data = road_data[road_data['Vehicle_ID'] == id].loc[:, ['deri_v', 'dx', 'dv', 'deri_a_clipped', 'v_l']]
        car_features = np.vstack((
            car_data['deri_v']/17-0.5,
            # car_data['v_l'] / 17 - 0.5,
            car_data['dv']/10,
            car_data['dx']/86-0.5,
                                  )).T
        car_targets = car_data['deri_a_clipped'].as_matrix()[:, np.newaxis]
        # car_targets = car_data['deri_v'].as_matrix()[:, np.newaxis]
        iter_start = 0
        zero_initial_state = True
        while True:
            if iter_start+TIME_STEPS+1 <= len(car_features):
                yield [car_features[iter_start:iter_start+TIME_STEPS, :][np.newaxis, :, :],
                       car_targets[(iter_start+1):(iter_start+TIME_STEPS+1), :][np.newaxis, :, :],  # predict next step
                       zero_initial_state]
                iter_start += TIME_STEPS
                zero_initial_state = False
            else:
                break


def ms_error(y_pre, y_target):
    return tf.square(tf.sub(y_pre, y_target))


def _weight_variable(shape, name='weights'):
    initializer = tf.random_normal_initializer(mean=0., stddev=1., )
    return tf.get_variable(shape=shape, initializer=initializer, name=name)


def _bias_variable(shape, name='biases'):
    initializer = tf.constant_initializer(0.1)
    return tf.get_variable(name=name, shape=shape, initializer=initializer)

with tf.variable_scope('inputs'):
    xs = tf.placeholder(tf.float32, [BATCH_SIZE, TIME_STEPS, INPUT_SIZE], name='xs')
    ys = tf.placeholder(tf.float32, [BATCH_SIZE, TIME_STEPS, OUTPUT_SIZE], name='ys')

with tf.variable_scope('input_layer'):
    l_in_x = tf.reshape(xs, [-1, INPUT_SIZE], name='2_2D')  # (batch*n_step, in_size)
    # Ws (in_size, cell_size)
    Wi = _weight_variable([INPUT_SIZE, CELL_SIZE])
    # bs (cell_size, )
    bi = _bias_variable([CELL_SIZE, ])
    # l_in_y = (batch * n_steps, cell_size)
    with tf.name_scope('Wx_plus_b'):
        l_in_y = tf.matmul(l_in_x, Wi) + bi
    with tf.name_scope('activation'):
        l_in_y = tf.nn.relu(l_in_y)
    # reshape l_in_y ==> (batch, n_steps, cell_size)
    l_in_y = tf.reshape(l_in_y, [-1, TIME_STEPS, CELL_SIZE], name='2_3D')


with tf.variable_scope('lstm_cell'):
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(CELL_SIZE, forget_bias=1.0, state_is_tuple=True)
    with tf.name_scope('initial_state'):
        cell_init_state = lstm_cell.zero_state(BATCH_SIZE, dtype=tf.float32)

    cell_outputs = []
    for t in range(TIME_STEPS):
        if t == 0:
            cell_output, cell_state = lstm_cell(l_in_y[:, t, :], cell_init_state)
        else:
            tf.get_variable_scope().reuse_variables()
            cell_output, cell_state = lstm_cell(l_in_y[:, t, :], cell_state)
        cell_outputs.append(cell_output)
    cell_final_state = cell_state

with tf.variable_scope('output_layer'):
    # cell_outputs_reshaped (BATCH*TIME_STEP, CELL_SIZE)
    cell_outputs_reshaped = tf.reshape(tf.concat(1, cell_outputs), [-1, CELL_SIZE])
    Wo = _weight_variable((CELL_SIZE, OUTPUT_SIZE))
    bo = _bias_variable((OUTPUT_SIZE,))
    pred = tf.matmul(cell_outputs_reshaped, Wo) + bo

with tf.name_scope('cost'):
    # compute cost for the cell_outputs
    losses = tf.nn.seq2seq.sequence_loss_by_example(
                [tf.reshape(pred, [-1], name='reshape_pred')],
                [tf.reshape(ys, [-1], name='reshape_target')],
                [tf.ones([BATCH_SIZE*TIME_STEPS], dtype=tf.float32)],
                average_across_timesteps=True,
                softmax_loss_function=ms_error,
                name='losses'
            )
    cost = tf.div(tf.reduce_sum(losses, name='losses_sum'), BATCH_SIZE,
                                    name='average_cost')

with tf.name_scope('trian'):
    train_op = tf.train.AdamOptimizer(0.001).minimize(cost)

sess = tf.Session()
sess.run(tf.initialize_all_variables())

tf.set_random_seed(1)
plt.ion()
fig, ax = plt.subplots(figsize=(13, 5))
plt.show()
# plt.ylim((-5, 20))  # 4 predicting speed
plt.ylim((-3.5, 3.5))  # 4 predicting acc
plt_time = 0
pred_to_plot = []
road_ys_to_plot = []
st = time.time()
for i in range(10000):
    if 'dg' not in globals():
        dg = data_gen()
    try:
        road_xs, road_ys, zero_initial_state = next(dg)
    except StopIteration:
        del dg
        continue
    if zero_initial_state:
        state_ = sess.run(cell_init_state)
    feed_dict = {xs: road_xs, ys: road_ys, cell_init_state: state_}
    _, state_, cost_, pred_ = sess.run([train_op, cell_final_state, cost, pred], feed_dict=feed_dict)
    pred_to_plot += pred_.flatten().tolist()
    road_ys_to_plot += road_ys.flatten().tolist()
    if zero_initial_state:
        init_time = plt_time
        init_acc = road_ys.flatten()[0]
    if i % 200 == 0:
        if len(pred_to_plot) > 600:
            pred_to_plot = pred_to_plot[-600:]
            road_ys_to_plot = road_ys_to_plot[-600:]
        if 'pred_line' not in globals():
            init_scatter = plt.scatter([init_time], [init_acc], s=100, c='red', alpha=.5, edgecolors=None)
            pred_line, = plt.plot(np.arange(plt_time, plt_time+TIME_STEPS), pred_to_plot, 'b-')
            road_line, = plt.plot(np.arange(plt_time, plt_time+TIME_STEPS), road_ys_to_plot, 'r-')
        else:
            init_scatter.set_offsets([init_time, init_acc])
            pred_line.set_data(np.arange(plt_time+TIME_STEPS-len(pred_to_plot), plt_time+TIME_STEPS), pred_to_plot)
            road_line.set_data(np.arange(plt_time+TIME_STEPS-len(pred_to_plot), plt_time+TIME_STEPS), road_ys_to_plot)
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.xlim((plt_time-600, plt_time+TIME_STEPS+10))
        plt.pause(0.0001)
        print(round(time.time() - st, 2))
        st = time.time()
    plt_time += TIME_STEPS

