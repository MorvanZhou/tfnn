import pandas as pd
import numpy as np


def select_lane(path, lane=1):
    df = pd.read_pickle(path)
    data = df.loc[:, ['Vehicle_ID', 'Frame_ID', 'Lane_Identification', 'Preceding_Vehicle', 'filter_position',
              'deri_v', 'deri_a_clipped', 'dx', 'dv', 'h', 'v_l']]
    # being lesser affected by lane changing
    lane_data = data[data.Lane_Identification == lane].reset_index(drop=True)
    lane_read = lane_data.copy()

    # lane1_data.to_csv('datasets/s.csv')
    # lane1_data = pd.read_csv('datesets/s.csv', index_col=0)
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
                data_to_be_append.set_value(i+40, dx)       # gap
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

    data = lane_data.loc[:, ['Vehicle_ID', 'filter_position']]
    all_ids = data['Vehicle_ID'].unique()
    delta_xs = pd.Series(name='displacement')
    for car_id in all_ids:
        p_data = data['filter_position'][data['Vehicle_ID'] == car_id]
        delta_x = p_data.diff()
        delta_xs = delta_xs.append(delta_x)
    lane_data.insert(7, 'delta_x', delta_xs)
    print(lane_data.head())
    # lane1_data.to_csv('datasets/s3.csv')
    lane_data.to_pickle(path[:17]+'_lane%s.pickle' % lane)


def combine_gap_displacement_data(all_paths):
    all_road_data = pd.read_pickle(all_paths[0]).iloc[:, :12]
    for path in all_paths[1:]:
        all_road_data = pd.concat((all_road_data, pd.read_pickle(path).iloc[:, :12]), axis=0)
    print(all_road_data.shape)
    all_road_data.rename(columns={'delta_x': 'displacement'}, inplace=True)
    all_road_data.to_pickle('datasets/I80-0400-0415-filter_0.8_gap_displacement.pickle')


def get_displacement(path):
    all_road_data = pd.read_pickle(path)
    all_car_id = np.unique(all_road_data['Vehicle_ID'])
    all_road_data['displacement'] = ''
    for i in all_car_id:
        single_car = all_road_data[all_road_data['Vehicle_ID'] == i]
        single_car_p = single_car['filter_position']
        single_car_index = single_car.index
        single_car_displacement = np.diff(single_car_p)
        all_road_data.iloc[single_car_index[1:], -1] = single_car_displacement
    all_road_data.to_pickle('datasets/I80-0500-0515-filter_0.8_T_v_ldxdvhdisplace.pickle')
    all_road_data.iloc[:5000, :].to_csv('datasets/I80-0500-0515-filter_sample.csv')


def get_proceeding_position(path):
    road_data = pd.read_pickle(path)
    road_data['proceeding_position'] = road_data['filter_position'] + road_data['dx']
    road_data.to_pickle('datasets/I80-0500-0515-filter_0.8_T_v_ldxdvhdisplace_proposition.pickle')
    road_data.iloc[:5000, :].to_csv('datasets/I80-0500-0515-filter_sample.csv')



if __name__ == '__main__':
    # path = 'datasets/I80-0400-0415-filter_0.8_T_v_ldxdvh.pickle'
    # select_lane(path, lane=5)
    # combine_gap_displacement_data([
    #     'datasets/I80-0400_lane1.pickle',
    #     'datasets/I80-0400_lane2.pickle',
    #     'datasets/I80-0400_lane3.pickle',
    #     'datasets/I80-0400_lane4.pickle',
    # ])
    # get_displacement('datasets/I80-0500-0515-filter_0.8_T_v_ldxdvh.pickle')
    get_proceeding_position('datasets/I80-0500-0515-filter_0.8_T_v_ldxdvhdisplace.pickle')

