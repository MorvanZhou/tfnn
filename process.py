import pandas as pd


def select_lane(path, lane=1):
    df = pd.read_pickle(path)
    data = df.loc[:, ['Vehicle_ID', 'Frame_ID', 'Lane_Identification', 'Preceding_Vehicle', 'filter_position',
              'deri_v', 'deri_a_clipped', 'dx', 'dv', 'h', 'v_l']]
    # being lesser affected by lane changing
    lane_data = data[data.Lane_Identification == lane].reset_index(drop=True)
    lane_read = lane_data.copy()

    # lane1_data.to_csv('datasets/s.csv')
    # lane1_data = pd.read_csv('datasets/s.csv', index_col=0)
    print(lane_data.head())
    all_time_data = pd.DataFrame()
    time_data = pd.DataFrame()
    for index, row in lane_data.iterrows():
        id = int(row.Vehicle_ID)
        f_id = int(row.Frame_ID)
        data_to_be_append = pd.Series(index=range(4*20))
        car_data = lane_read[lane_read.Vehicle_ID == id]
        for t, i in zip(range(f_id-20, f_id), range(20)):
            if car_data.Frame_ID.isin([t]).any():
                car_data_at_t = car_data[car_data.Frame_ID == t]
                v = car_data_at_t.deri_v
                v_l = car_data_at_t.v_l
                dx = car_data_at_t.dx
                dv = car_data_at_t.dv
                data_to_be_append.set_value(i, v)
                data_to_be_append.set_value(i+20, v_l)
                data_to_be_append.set_value(i+40, dx)
                data_to_be_append.set_value(i+60, dv)

        time_data = time_data.append(data_to_be_append, ignore_index=True)
        if index % 500 == 0 or index == lane_data.index.max():
            all_time_data = pd.concat([all_time_data, time_data], axis=0, ignore_index=True)
            time_data = pd.DataFrame()
        if index % 200 == 0:
            print(index/lane_data.shape[0])

    cols = ['v_%s' % i for i in range(20)] + ['v_l_%s' % i for i in range(20)] + \
           ['dx_%s' % i for i in range(20)] + ['dv_%s' % i for i in range(20)]
    old_cols = list(range(4*20))
    all_time_data.rename(columns=dict(zip(old_cols, cols)), inplace=True)
    lane_data = pd.concat([lane_data, all_time_data], axis=1)
    print(lane_data.head())
    # lane1_data.to_csv('datasets/s3.csv')
    lane_data.to_pickle(path[:17]+'_lane%s.pickle' % lane)

if __name__ == '__main__':
    path = 'datasets/I80-0500-0515-filter_0.8_T_v_ldxdvh.pickle'
    select_lane(path, lane=5)
