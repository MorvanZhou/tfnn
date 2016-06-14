import pandas as pd
import numpy as np


def data_organize(path, lane):
    df = pd.read_pickle(path)
    # load and select lane
    load_all_data = df[df['lane_id'] == lane].reset_index(drop=True)
    # filter the data we don't need
    load_data = load_all_data.loc[:, ['id', 'f_id', 'leader_id', 'a', 'v', 's']]
    # get last 1 second data for every car
    total_samples = load_data.shape[0]

    to_passed_second = 2
    # edited_data has [vs, v_ls, ss]
    col_vs = ['v(-%s)' % round(to_passed_second-i/10, 1) for i in range(1, to_passed_second*10+1)]
    col_vs_l = ['v_l(-%s)' % round(to_passed_second-i/10, 1) for i in range(1, to_passed_second*10+1)]
    col_ss = ['s(-%s)' % round(to_passed_second-i/10, 1) for i in range(1, to_passed_second*10+1)]
    cols_for_edited = ['id', 'f_id', 'leader_id', 'a']+col_vs+col_vs_l+col_ss
    edited_data = pd.DataFrame(columns=cols_for_edited)

    for index, row in load_data.iterrows():
        basic_info = [row.id, row.f_id, row.leader_id, row.a]
        if index < to_passed_second*10-1:
            edited_data.loc[index] = basic_info+[np.nan]*60
            continue
        f_id = row.f_id  # f_id
        leader = row.leader_id  # leader
        vs = []
        leader_vs = []
        ss = []
        for i in range(to_passed_second*10):
            try:
                # check and try find leader's data in last f_id
                leader_v = load_data.v.loc[(load_data.f_id == f_id - i) & (load_data.id == leader)].values[0]
            except IndexError:
                leader_v = np.nan
            # insert leader v to the first position
            leader_vs.insert(0, leader_v)
            if load_data.loc[index, 'id'] == load_data.loc[index-i, 'id']:
                v = load_data.loc[index - i, 'v']
                s = load_data.loc[index - i, 's']

            else:
                v, s = np.nan, np.nan
            # insert v and s
            vs.insert(0, v)
            ss.insert(0, s)

        append_results = basic_info+vs+leader_vs+ss
        edited_data.loc[index] = append_results
        if index % 1000 == 0:
            print(round(index/total_samples*100, 2), '%')

    edited_data.to_pickle(path+'-lane'+str(lane)+'pickle')


def convert_from_txt(path, save_example=True, road='I80'):
    df = pd.read_csv(path, delim_whitespace=True, names=['id', 'f_id', 'total_f', 'GT', 'local_x',
                                                        'local_y', 'Global_X', 'Global_Y', 'l', 'w', 'type',
                                                        'v', 'a', 'lane_id', 'leader_id', 'follower_id',
                                                        's', 'h'
                                                        ])
    # convert from foot to meter
    df[['local_x', 'local_y', 'Global_X', 'Global_Y', 'l', 'w', 'v', 'a', 's']] = \
        df[['local_x', 'local_y', 'Global_X', 'Global_Y', 'l', 'w', 'v', 'a', 's']] * 0.3048
    file_name = path.split('/')[-1].split('.')[0]
    df.to_pickle('/home/morvan/Documents/python/tfnn/road data/'+road+'/'+file_name+'.pickle')
    if save_example is True:
        df.iloc[:10000, :].to_csv('/home/morvan/Documents/python/tfnn/road data/'+road+'/'+file_name+'.csv', index=False)


def filter_second(path, second,):
    if not 0 < second <= 2:
        raise ValueError('second should between 0 and 2 second')
    df = pd.read_pickle(path)
    mini_second = int(second*10)
    selected_v = df.iloc[:, (24-mini_second):24]
    selected_v_l = df.iloc[:, 44-mini_second:44]
    selected_s = df.iloc[:, 64-mini_second:64]
    acceleration = df.iloc[:, 3]
    train_data = pd.concat((acceleration, selected_v, selected_v_l, selected_s), axis=1, copy=True)
    # drop np.nan data
    train_data_dropped = train_data.dropna(axis=0, how='any')
    # drop spacing > 200 m
    print(np.amax(train_data_dropped.iloc[:,-10:]))
    train_data_dropped.to_pickle('train_I80_lane1.pickle')

if __name__ == '__main__':
    # convert_from_txt('trajectories-0820am-0835am.txt', save_example=False, road='US101')
    data_organize('road data/I80/trajectories-0400-0415.pickle', lane=4)
    # filter_second('road data/I80/0400-0415-lane1.pickle', 1)