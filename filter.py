import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
# plt.style.use('ggplot')


def extract(path='datasets/I80-0400-0415.txt', to_pickle=True):
    columns_name = \
        ['Vehicle_ID', 'Frame_ID', 'Total_Frames', 'Global_Time', 'Local_X', 'Local_Y',
         'Global_X', 'Global_Y', 'Vehicle_Length', 'Vehicle_Width', 'Vehicle_Class', 'Vehicle_Velocity',
         'Vehicle_Acceleration', 'Lane_Identification', 'Preceding_Vehicle', 'Following_Vehicle', 'Spacing',
         'Headway']
    df = pd.read_csv(path, delim_whitespace=True, names=columns_name)
    df.loc[:, ['Local_X', 'Local_Y', 'Global_X', 'Global_Y', 'Vehicle_Length', 'Vehicle_Width', 'Vehicle_Velocity',
            'Vehicle_Acceleration', 'Spacing']] *= 0.3048
    if to_pickle:
        df.to_pickle(path[:-3]+'pickle')
    return df


def filter_x(data, alpha, T, dt=0.1):
    T_x = T
    delta_x = T_x / dt
    vehicle = data.loc[data['Vehicle_ID'] == alpha]
    # N_alpha check
    N_alpha_1 = len(vehicle.Total_Frames)
    N_alpha_2 = vehicle.Total_Frames.iloc[0]
    N_alpha = N_alpha_1 if N_alpha_1 == N_alpha_2 else print('N_alpha error')
    positions = vehicle['Local_Y']

    time_points = pd.Series(np.arange(1, N_alpha + 1, dtype=np.int))
    D1_x = pd.Series(np.ones((N_alpha,)) * 3 * delta_x)

    D2 = time_points - 1
    D3 = N_alpha - time_points
    D_all_x = pd.concat([D1_x, D2, D3], axis=1).astype(np.int)

    D_x = D_all_x.min(axis=1)

    time_ranges_x = pd.concat([time_points - D_x, time_points + D_x], axis=1)
    time_ranges_x = time_ranges_x.astype(np.int)

    results = pd.Series(name='filter_position')
    for i, t_range_x in zip(time_points, time_ranges_x.values):
        k_all_x = np.arange(t_range_x[0], t_range_x[1] + 1, dtype=np.int)
        z_results_x = np.exp(-np.abs(i - k_all_x) / delta_x)
        Z_x = np.sum(z_results_x)
        positions_results = positions.iloc[k_all_x - 1] * z_results_x
        result_sample = 1 / Z_x * np.sum(positions_results)
        results.set_value(len(results), result_sample)


    position = results
    x1 = position.shift(-2)
    x0 = position.copy()
    v = (x1 - x0) / .2
    v = v.shift(1)

    v0 = v.copy()
    v1 = v.shift(-2)
    a = (v1 - v0) / .2
    a = a.shift(1)
    v = v.rename('deri_v')
    a = a.rename('deri_a')

    # ignore the value near boundaries due to large biases.
    v.iloc[: int(3 * delta_x)] = np.nan
    v.iloc[-int(3 * delta_x):] = np.nan
    a.iloc[: int(3 * delta_x)] = np.nan
    a.iloc[-int(3 * delta_x):] = np.nan

    results = pd.concat([results, v, a], axis=1)
    results.index = positions.index
    return results


def filter_x_v_a(data, alpha, T, dt=0.1):
    T_x, T_v, T_a = T, 1, 4
    delta_x, delta_v, delta_a = T_x/dt, T_v/dt, T_a/dt
    vehicle = data.loc[data['Vehicle_ID'] == alpha]
    # N_alpha check
    N_alpha_1 = len(vehicle.Total_Frames)
    N_alpha_2 = vehicle.Total_Frames.iloc[0]
    N_alpha = N_alpha_1 if N_alpha_1 == N_alpha_2 else print('N_alpha error')
    positions = vehicle['Local_Y']
    velocities = vehicle['Vehicle_Velocity']
    accelerations = vehicle['Vehicle_Acceleration']

    time_points = pd.Series(np.arange(1, N_alpha+1))
    D1_x = pd.Series(np.ones((N_alpha,)) * 3 * delta_x)
    D1_v = pd.Series(np.ones((N_alpha,)) * 3 * delta_v)
    D1_a = pd.Series(np.ones((N_alpha,)) * 3 * delta_a)
    D2 = time_points - 1
    D3 = N_alpha - time_points
    D_all_x = pd.concat([D1_x, D2, D3], axis=1)
    D_all_v = pd.concat([D1_v, D2, D3], axis=1)
    D_all_a = pd.concat([D1_a, D2, D3], axis=1)
    D_x = D_all_x.min(axis=1)
    D_v = D_all_v.min(axis=1)
    D_a = D_all_a.min(axis=1)
    time_ranges_x = pd.concat([time_points - D_x, time_points + D_x], axis=1)
    time_ranges_v = pd.concat([time_points - D_v, time_points + D_v], axis=1)
    time_ranges_a = pd.concat([time_points - D_a, time_points + D_a], axis=1)

    results = pd.DataFrame(columns=['filter_position', 'filter_velocity', 'filter_acceleration'])
    for i, t_range_x, t_range_v, t_range_a \
            in zip(time_points, time_ranges_x.values, time_ranges_v.values, time_ranges_a.values):
        k_all_x = np.arange(t_range_x[0], t_range_x[1]+1)
        k_all_v = np.arange(t_range_v[0], t_range_v[1] + 1)
        k_all_a = np.arange(t_range_a[0], t_range_a[1] + 1)
        z_results_x = np.exp(-np.abs(i-k_all_x)/delta_x)
        z_results_v = np.exp(-np.abs(i - k_all_v) / delta_v)
        z_results_a = np.exp(-np.abs(i - k_all_a) / delta_a)
        Z_x = np.sum(z_results_x)
        Z_v = np.sum(z_results_v)
        Z_a = np.sum(z_results_a)
        positions_results = positions.iloc[k_all_x-1]*z_results_x
        velocities_results = velocities.iloc[k_all_v-1]*z_results_v
        accelerations_results = accelerations.iloc[k_all_a - 1] * z_results_a
        result_sample = pd.Series([1/Z_x * np.sum(positions_results),
                                   1/Z_v * np.sum(velocities_results),
                                   1/Z_a * np.sum(accelerations_results)],
                                  index=['filter_position', 'filter_velocity', 'filter_acceleration'])
        results = results.append(result_sample, ignore_index=True)

    results.index = positions.index

    # accelerations.plot(c='r')
    # results['filter_acceleration'].plot(c='b')
    # velocities.plot(c='r')
    # results['filter_velocity'].plot(c='b')
    # plt.show()
    return results


def batch_filter(path, T=0.8, selected_filter='f_x', save=True, clip_bound=None):
    data = pd.read_pickle(path)
    all_vehicle_id = data['Vehicle_ID'].unique()
    filtered_data = pd.DataFrame()
    for i, id in enumerate(all_vehicle_id):
        if selected_filter == 'f_x':
            result = filter_x(data, alpha=id, T=T)
        elif selected_filter == 'f_x_v_a':
            result = filter_x_v_a(data, id, T)
        filtered_data = pd.concat([filtered_data, result], axis=0)
        print(round(i/len(all_vehicle_id)*100, 1))
    if clip_bound is not None:
        filtered_data['deri_a_clipped'] = filtered_data['deri_a'].clip(clip_bound[0], clip_bound[1])

    c_data = pd.concat([data, filtered_data], axis=1)
    c_data.to_pickle(path[:22] + '-filter_%s_T.pickle' % str(T))
    # data.dropna(how='any', inplace=True)
    # print(data.head())
    if save:
        extract_v_l_dx_dv_h(path[:22] + '-filter_%s_T.pickle' % str(T), save=True)
    else:
        return c_data


def plot_comparison(data, alpha, which):
    vehicle = data[data['Vehicle_ID'] == alpha]
    if which == 'p':
        plt.plot(vehicle['Local_Y'])
        plt.plot(vehicle['filter_position'])
        plt.ylabel('$Position$')
        plt.legend(loc='upper right')
    elif which == 'a':
        plt.plot(vehicle['Vehicle_Acceleration'].iloc[40:250], '-k', label='$Unfiltered$')
        plt.plot(vehicle['deri_a_clipped'].iloc[40:250], '--k', label='$Filtered$')
        plt.ylabel('$Acceleration$')
        plt.legend(loc='upper right')
    elif which == 'v':
        plt.plot(vehicle['Vehicle_Velocity'].iloc[40:250],'-k', label='$Unfiltered$')
        plt.plot(vehicle['deri_v'].iloc[40:250], '--k',label='$Filtered$')
        plt.ylabel('$Velocity$')
        plt.legend(loc='upper right')
    elif which == 'h':
        vehicle['Headway'].plot(c='r')
        vehicle['filter_headway'].plot(c='b')
        plt.ylabel('headway')
        plt.legend(loc='best')
    elif which == 'sc':
        vehicle['Spacing'].plot(c='r')
        vehicle['filter_spacing'].plot(c='b')
        plt.ylabel('spacing')
        plt.legend(loc='best')
    elif which == 'a and v':
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        ax1.plot(vehicle['Vehicle_Acceleration'].iloc[40:250], '--k', label='$Unfiltered$')
        ax1.plot(vehicle['deri_a_clipped'].iloc[40:250], '-k', label='$Filtered$')
        ax1.set_ylabel('$Acceleration\ (m/s^2)$')
        ax1.set_xticks([], [])
        ax1.set_xlabel('$Time$')
        ax1.set_xlim((6500, 6650))
        ax1.legend(loc='lower right')

        ax2.plot(vehicle['Vehicle_Velocity'].iloc[40:250], '--k', label='$Unfiltered$')
        ax2.plot(vehicle['deri_v'].iloc[40:250], '-k', label='$Filtered$')
        ax2.set_ylabel('$Velocity\ (m/s)$')
        ax2.set_xlabel('$Time$')
        ax2.set_xlim((6500, 6650))
        ax2.set_xticks([], [])
        ax2.legend(loc='lower right')
    # plt.savefig('comparison_filtered_unfiltered.png', format='png', dpi=700,)
    plt.show()


def differentiation(data, alpha):
    vehicle = data.loc[data['Vehicle_ID'] == alpha]
    # choose to plot filtered result
    position = vehicle['filter_position']
    # choose to plot original result
    position = vehicle['Local_Y']
    x1 = position.shift(-2)
    x0 = position.copy()
    v = (x1 - x0) / .2
    v = v.shift(1)

    v0 = v.copy()
    v1 = v.shift(-2)
    a = (v1 - v0) / .2
    a = a.shift(1)

    v = v.rename('1st order diff equ of unfiltered Y')
    a = a.rename('2nd order diff equ of unfiltered Y')
    start = 5
    plt.figure(0)
    vehicle['Local_Y'].plot(c='r')
    vehicle['filter_position'].iloc[start:].plot(c='b')
    plt.xlabel('time')
    plt.ylabel('position')
    plt.legend(loc='best')

    plt.figure(1)
    vehicle['Vehicle_Velocity'].plot(c='r')
    vehicle['deri_v'].iloc[start:].plot(c='g')
    v.plot(c='b')
    plt.xlabel('time')
    plt.ylabel('velocity')
    plt.legend()

    plt.figure(2)
    vehicle['Vehicle_Acceleration'].plot(c='r')
    a.plot(c='b')
    vehicle['deri_a_clipped'].iloc[start:].plot(c='g')
    plt.legend()
    plt.xlabel('time')
    plt.ylabel('acceleration')
    plt.show()


def select_T(data):
    validation = pd.Series()
    for T in np.arange(0.1, 1.6, 0.2):
        f_data = batch_filter(data.copy(), T, selected_filter='f_x', save=False)
        percentage_unexpected_a = (np.abs(f_data.deri_a) > 3.41376).sum()/len(f_data.deri_a)
        validation = validation.set_value(T, percentage_unexpected_a)
        print('T:', T, 'percentage:', percentage_unexpected_a)
        del f_data
    validation.plot()
    plt.xlabel('T value')
    plt.ylabel('Percentage of acceleration that exceeds the threshold')
    plt.show()


def extract_v_l_dx_dv_h(path, save=False):
    t0 = time.time()
    raw_data = pd.read_pickle(path)
    data = raw_data.loc[:, ['Vehicle_ID', 'Frame_ID', 'Preceding_Vehicle',
                                        'filter_position', 'deri_v']]
    read_data = data.copy()
    bkup_data = data.copy()
    wrt_all_data = pd.DataFrame(columns=['v_l', 'dx', 'dv', 'h'])
    for index, row in bkup_data.iterrows():
        f_id = row.Frame_ID
        pre_car_id = row.Preceding_Vehicle
        if index % 10000 == 0:
            if index == 0:
                wrt_data = pd.DataFrame(columns=['v_l', 'dx', 'dv', 'h'])
            else:
                wrt_all_data = pd.concat([wrt_all_data, wrt_data], axis=0, ignore_index=True)
                wrt_data = pd.DataFrame(columns=['v_l', 'dx', 'dv', 'h'])
            if index >= len(read_data) - 10000:
                partial_data = read_data.iloc[index-10000:, :]
            elif index < 10000:
                partial_data = read_data.iloc[:20000, :]
            else:
                partial_data = read_data.iloc[index-10000:index+10000, :]

        pre_car = partial_data.loc[(partial_data.Frame_ID == f_id) & (partial_data.Vehicle_ID == pre_car_id)]
        # read_data.drop(pre_car.index, inplace=True)
        dx = pre_car.filter_position - row.filter_position
        v_l = pre_car.deri_v
        dv = pre_car.deri_v - row.deri_v
        h = dx/row.deri_v
        v_l = np.nan if len(v_l) == 0 else v_l.iloc[0]
        dx = np.nan if len(dx) == 0 else dx.iloc[0]
        dv = np.nan if len(dv) == 0 else dv.iloc[0]
        h = np.nan if len(h) == 0 else h.iloc[0]
        to_be_wrt = {'v_l': v_l, 'dx': dx, 'dv': dv, 'h': h}
        wrt_data = wrt_data.append(to_be_wrt, ignore_index=True)
        if index % 100 == 0:
            percent = round(index / len(bkup_data), 4) * 100
            t1 = time.time()
            time_spend = t1-t0
            time_remains = (time_spend/(index / len(bkup_data)))/60   # min
            print(round(time_remains, 2), ' mins',  str(percent)+'%', row.Vehicle_ID, to_be_wrt)
        # if index > 1000:
        #     # check
        #     pro_data = pd.concat([data, wrt_data], axis=1)
        #     pro_data.iloc[:2000,: ].to_csv(path[:-7] + '_dxdvh.csv')
        #     break
    pro_data = pd.concat([raw_data, wrt_all_data], axis=1)
    if save:
        pro_data.to_pickle(path[:-7]+'_v_ldxdvh.pickle')
    return pro_data


if __name__ == '__main__':
    # path = 'datasets/I80-0500-0515.txt'
    # extract(path=path)

    # select 3000-3300 car example
    # data = pd.read_pickle('datasets/I80-0400-0415.pickle').iloc[1096646:1236558, :]
    # select_T(data)
    # filter_x(data, alpha=17, T=1.5)
    # filter_x_v_a(data, 4)

    # path = 'datasets/I80-0500-0515.pickle'
    # batch_filter(path, T=0.8, selected_filter='f_x', clip_bound=(-3.41376, 3.41376), save=True)

    path = 'datasets/I80-0400-0415-filter_0.8_T.pickle'
    data = pd.read_pickle(path)
    plot_comparison(data, alpha=17, which='a and v')
    # differentiation(data,4)

    # path = 'datasets/I80-0400-0415-filter_0.8_T.pickle'
    # extract_v_l_dx_dv_h(path, save=True)
