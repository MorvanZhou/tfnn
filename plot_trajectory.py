import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ColorLine import colorline


def plot_traj(df1):
    all_v_id = np.unique(df1.Vehicle_ID)
    fig = plt.figure()
    for id in all_v_id:
        vehicle = df1[df1.Vehicle_ID == id].loc[:, ['filter_position', 'deri_v', 'Frame_ID']].dropna()
        traj = vehicle.filter_position
        speed = vehicle.deri_v/15
        t = (vehicle.Frame_ID/10).astype(np.int)
        # traj = vehicle.Local_Y
        colorline(t.values, traj.values, speed.values, linewidth=0.7, cmap='Spectral')
    plt.xlim(0, 17*60)
    plt.ylim(0, 500)

    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.yaxis.tick_left()
    ax.xaxis.tick_bottom()
    plt.ylabel('$Distance (m)$')
    plt.xticks([0, 333, 666, 1000], ['4:00', '4:05', '4:10', '4:15'])
    plt.xlabel('$Time (min)$')

    plt.text(-100, 520, '$(b)$', fontdict={'size': 20, })       # numbering
    label = plt.text(850, 50, r'$Speed\ km/h$',
             fontdict={'size': 12, 'weight': 'bold',})          # colorbar
    label.set_bbox(dict(facecolor='white', edgecolor='None', alpha=0.7))
    cbaxes = fig.add_axes([0.67, 0.15, 0.2, 0.02])
    sm = plt.cm.ScalarMappable(cmap='Spectral', norm=plt.Normalize(vmin=0, vmax=60))
    sm._A = []
    cbar = plt.colorbar(sm, ticks=[0, 30, 60], cax=cbaxes, orientation='horizontal')
    cbar.ax.set_xticklabels(['$0$', '$30$', '$60$'], fontdict={'size': 12})
    plt.savefig('I80-0400-0415_lane3.png', format='png', dpi=500)
    # plt.show()


def plot_data(df, id=2000):
    vehicle = df[df.Vehicle_ID == id]
    legend_size = 10
    fig, axes = plt.subplots(nrows=4, ncols=1)
    vehicle.deri_a_clipped.plot(ax=axes[0], c='b')
    vehicle.Vehicle_Acceleration.plot(ax=axes[0], c='r')
    axes[0].set_ylabel('acc')
    axes[0].legend(prop={'size':legend_size})

    vehicle.deri_v.plot(ax=axes[1], c='b')
    vehicle.Vehicle_Velocity.plot(ax=axes[1], c='r')
    axes[1].legend(prop={'size':legend_size})
    axes[1].set_ylabel('speed')

    vehicle.dx.plot(ax=axes[2], c='b')
    vehicle.Spacing.plot(ax=axes[2], c='r')
    axes[2].legend(prop={'size':legend_size})
    axes[2].set_ylabel('dx')

    vehicle.h.plot(ax=axes[3], c='b')
    vehicle.Headway.plot(ax=axes[3], c='r')
    axes[3].legend(prop={'size':legend_size})
    axes[3].set_ylabel('headway')

    plt.show()

df = pd.read_pickle('datasets/I80-0400-0415-filter_0.8_T_v_ldxdvh.pickle')
df1 = df[df.Lane_Identification == 3]
# df2 = pd.read_pickle('datasets/I80-0400-0415.pickle')
# df2 = df2[df2.Lane_Identification == 2]
plot_traj(df1)
# plot_data(df1, id=1050)

