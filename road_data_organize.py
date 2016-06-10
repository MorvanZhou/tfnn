import pandas as pd
import numpy as np


def data_organize():
    # get last 1 second data for every car
    load_data = pd.read_csv('tmp0.csv')
    total_samples = load_data.shape[0]
    edited_data = load_data.copy()
    for i in range(1, 10):
        edited_data['v%i' % i] = np.nan
    for i in range(10):
        edited_data['v_l%i' % i] = np.nan
    for i in range(1, 10):
        edited_data['s%i' % i] = np.nan
    data_array = load_data.as_matrix()
    for index, row in enumerate(data_array):
        f_id = row[1]  # f_id
        leader = row[6]  # leader
        vs = []
        leader_vs = []
        ss = []
        for i in range(0, 10):
            if index <= 10:
                is_break = True
                break
            if not (np.any((edited_data.f_id == f_id - i) & (edited_data.id == leader)) == True):
                is_break = True
                break
            is_break = False
            leader_v = edited_data.v.loc[(edited_data.f_id == f_id - i) & (edited_data.id == leader)].values[0]
            leader_vs.append(leader_v)
            if i >= 1:
                v = data_array[index - i][10]
                vs.append(v)
                s = data_array[index - i][8]
                ss.append(s)
        if not is_break:
            edited_data.iloc[index, -9:] = ss
            edited_data.iloc[index, -19:-9] = leader_vs
            edited_data.iloc[index, -28:-19] = vs
        if index % 1000 == 0:
            print(round(index/total_samples*100, 2), '%')

    edited_data.dropna(axis=0, inplace=True)
    edited_data.to_csv('tmp_organize.csv', index=False)


def to_training_data():
    df = pd.read_pickle('vehicle_data.pickle')
    train_dv = df['v_l'] - df['v']
    train_data = df[['a', 'v', 'dx', 'h', 'v_l']]
    train_h_copy = train_data.copy().h
    train_data.h[train_h_copy > 50] = 50
    train_data.dx[train_data.dx < 0] = 0
    train_data.dx[train_data.dx > 50] = 50
    train_data['dv'] = train_dv
    train_data.head(20000).to_csv('example_training_data.csv')
    train_data.to_pickle('training_data.pickle')


def pick_lane(lane):
    df = pd.read_pickle('trajectories-0750am-0805am.pickle')
    lane_data = df[df['Lane_Identification'] == lane]
    lane_data.to_csv('selected_lane_data.csv')

if __name__ == '__main__':
    data_organize()
    # pick_lane(1)
