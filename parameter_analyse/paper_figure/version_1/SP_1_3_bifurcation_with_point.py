import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.io.matlab import loadmat
from parameter_analyse.analyse_dynamic.print_figure.print_stability import print_stability


## path of the data
path_init = os.path.dirname(os.path.realpath(__file__)) + "/../../static/simulation/master_seed_0/"
path = os.path.dirname(__file__) + '/../../analyse_dynamic/matlab/'

## parameter of the figures
labelticks_size = 12
label_legend_size = 12
ticks_size = 12

# load bifurcation
b_30 = loadmat(path + '/b_30/EQ_Low/EQ_Low.mat', chars_as_strings=True, simplify_cells=True)
b_30['x'][:2] *= 1e3
b_30['x'][-1] *= 1e3
b_30['x'][2:5] *= 1e6

b_60 = loadmat(path + '/b_60/EQ_Low/EQ_Low.mat', chars_as_strings=True, simplify_cells=True)
b_60['x'][:2] *= 1e3
b_60['x'][-1] *= 1e3
b_60['x'][2:5] *= 1e6

b_0 = loadmat(path + '/EQ_Low/EQ_low.mat', chars_as_strings=True, simplify_cells=True)
b_0['x'][:2] *= 1e3
b_0['x'][-1] *= 1e3
b_0['x'][2:5] *= 1e6
b_0['x'] = np.vstack((b_0['x'][:5], np.zeros((1, b_0['x'].shape[1]), ), b_0['x'][5]))
# b_0['f'] = np.vstack((b_0['f'], np.zeros((1, b_0['f'].shape[1]))))

# get network result
network_0 = np.concatenate([np.expand_dims(range(100), axis=1), np.load(path_init + '/0.0_mean_var.npy')], axis=1)
network_30 = np.concatenate([np.expand_dims(range(100), axis=1), np.load(path_init + '/30.0_mean_var.npy')], axis=1)
network_60 = np.concatenate([np.expand_dims(range(100), axis=1), np.load(path_init + '/60.0_mean_var.npy')], axis=1)

for network, b, xmin, xmax, xticks, \
    y_ex_min,   y_ex_max,   y_ex_ticks, \
    y_ex_z_min, y_ex_z_max, y_ex_z_ticks, \
    y_ex_c_min, y_ex_c_max, y_ex_c_ticks, \
    y_in_min,   y_in_max,   y_in_ticks, \
    y_in_z_min, y_in_z_max, y_in_z_ticks, \
    y_in_c_min, y_in_c_max, y_in_c_ticks, \
    y_ad_min,   y_ad_max,   y_ad_ticks, \
    y_ad_z_min, y_ad_z_max, y_ad_z_ticks, \
    y_ei_c_min, y_ei_c_max, y_ei_c_ticks, \
    color, index \
        in \
        [(network_0,   b_0,
          -20.0, 100.0, [0.0, 50.0, 100.0],
          -0.1,  200.0, [0.0, 100.0, 200.0],
          -0.1,   30.0, [0.0, 10.0, 20.0],
          -0.1,    2.0, [0.0, 1.0, 2.0],
          -0.1,  200.0, [0.0, 100.0, 200.0],
          -0.1,  180.0, [0.0, 90.0, 180.0],
          -0.1,   30.0, [0.0, 15.0, 30.0],
          -0.1,  200.0, [0.0, 100.0, 200.0],
          -0.1,  180.0, [0.0, 90.0, 180.0],
          -15.0,  15.0, [-15.0, 0.0, 15.0],
          'k', 1),
         (network_30, b_30,
          -1.0, 120.0, [0.0, 60.0, 120.0],
          -0.1,  200.0, [0.0, 100.0, 200.0],
          -0.1,   30.0, [0.0, 10.0, 20.0],
          -0.1,    0.8, [0.0, 0.4, 0.8],
          -0.1,  200.0, [0.0, 100.0, 200.0],
          -0.1,  110.0, [0.0, 55.0, 110.0],
          -0.1,   22.0, [0.0, 10.0, 20.0],
          -0.1,  5.0, [0.0, 2.5, 5.0],
          -0.1,  180.0, [0.0, 90.0, 180.0],
          -10.0,  10.0, [-10.0, 0.0, 10.0],
          'b', 2),
         (network_60, b_60,
          0.0, 140.0, [0.0, 70.0, 140.0],
          -0.1,  200.0, [0.0, 100.0, 200.0],
          -0.1,   10.0, [0.0, 5.0, 10.0],
          -0.1,    1.0, [0.0, 0.5, 1.0],
          -0.1,  200.0, [0.0, 100.0, 200.0],
          -0.1,  100.0, [0.0, 50.0, 100.0],
          -0.1,   20.0, [0.0, 10.0, 20.0],
          -0.1,  10.0, [0.0, 5.0, 10.0],
          -0.1,  10.0, [0.0, 5.0, 10.0],
          -10.0,  10.0, [-7.0, 0.0, 7.0],
          'g', 3)]:
    ## make figure
    fig, axs = plt.subplots(3, 3, figsize=(6.8, 5.5))
    # excitatory
    plt.sca(axs[0, 0])
    print_stability(b['x'], b['f'], b['s'], 6, 0, color=color, letter=True, linewidth=1.0, ymin=y_ex_min, ymax=y_ex_max,
                    xmin=xmin, xmax=xmax)
    plt.plot(network[:, 0], network[:, 1], 'x', color=color, ms=5.0)
    # plt.vlines(0.0, ymin=-30.0, ymax=200.0, color='m')
    plt.xticks(xticks)
    # plt.xlabel("external input (Hz)", {"fontsize": labelticks_size})
    plt.yticks(y_ex_ticks)
    plt.ylabel("EXCITATORY\nfiring rate\npopulation (Hz)", {"fontsize": labelticks_size})
    plt.tick_params(labelsize=ticks_size)
    plt.annotate('A', xy=(-0.15, 0.83), xycoords='axes fraction', weight='bold', fontsize=labelticks_size)

    # excitatory zoom
    plt.sca(axs[0, 1])
    print_stability(b['x'], b['f'], b['s'], 6, 0, color=color, letter=True, linewidth=1.0, ymin=y_ex_z_min,
                    ymax=y_ex_z_max, xmin=xmin, xmax=xmax)
    plt.plot(network[:, 0], network[:, 1], 'x', color=color, ms=5.0)
    # plt.vlines(0.0, ymin=-30.0, ymax=200.0, color='m')
    plt.xticks(xticks)
    # plt.xlabel("external input", {"fontsize": labelticks_size})
    plt.yticks(y_ex_z_ticks)
    # plt.ylabel("firing rate of excitatory population (Hz)", {"fontsize": labelticks_size})
    plt.tick_params(labelsize=ticks_size)
    plt.annotate('B', xy=(-0.15, 0.83), xycoords='axes fraction', weight='bold', fontsize=labelticks_size)

    # variance excitatory
    plt.sca(axs[0, 2])
    print_stability(b['x'], b['f'], b['s'], 6, 2, color=color, letter=True, linewidth=1.0,
                    ymin=y_ex_c_min, ymax=y_ex_c_max, xmin=xmin, xmax=xmax)
    plt.plot(network[:, 0], network[:, 2], 'x', color=color, ms=5.0)
    # plt.vlines(0.0, ymin=-30.0, ymax=200.0, color='m')
    plt.xticks(xticks)
    # plt.xlabel("external input", {"fontsize": labelticks_size})
    plt.yticks(y_ex_c_ticks)
    # plt.ylabel("firing rate of excitatory\npopulation (Hz)", {"fontsize": labelticks_size})
    plt.tick_params(labelsize=ticks_size)
    plt.title("VARIANCE", {"fontsize": labelticks_size})
    plt.annotate('C', xy=(-0.15, 0.83), xycoords='axes fraction', weight='bold', fontsize=labelticks_size)

    # inhibitory
    plt.sca(axs[1, 0])
    print_stability(b['x'], b['f'], b['s'], 6, 1, color=color, letter=True, linewidth=1.0,
                    ymin=y_in_min, ymax=y_in_max, xmin=xmin, xmax=xmax)
    plt.plot(network[:, 0], network[:, 3], 'x', color=color, ms=5.0)
    # plt.vlines(0.0, ymin=-30.0, ymax=200.0, color='m')
    plt.xticks(xticks)
    # plt.xlabel("external input", {"fontsize": labelticks_size})
    plt.yticks(y_in_ticks)
    plt.ylabel("INHIBITORY\nfiring rate\npopulation (Hz)", {"fontsize": labelticks_size})
    plt.tick_params(labelsize=ticks_size)
    plt.annotate('D', xy=(-0.15, 0.83), xycoords='axes fraction', weight='bold', fontsize=labelticks_size)

    # inhibitory zoom
    plt.sca(axs[1, 1])
    print_stability(b['x'], b['f'], b['s'], 6, 1, color=color, letter=True, linewidth=1.0,
                    ymin=y_in_z_min, ymax=y_in_z_max, xmin=xmin, xmax=xmax)
    plt.plot(network[:, 0], network[:, 3], 'x', color=color, ms=5.0, label='b=0')
    # plt.vlines(0.0, ymin=-30.0, ymax=200.0, color='m')
    plt.xticks(xticks)
    plt.xlabel("external input (Hz)", {"fontsize": labelticks_size})
    plt.yticks(y_in_z_ticks)
    # plt.ylabel("firing rate of excitatory population (Hz)", {"fontsize": labelticks_size})
    plt.tick_params(labelsize=ticks_size)
    plt.annotate('E', xy=(-0.15, 0.83), xycoords='axes fraction', weight='bold', fontsize=labelticks_size)

    # variance inhibitory
    plt.sca(axs[1, 2])
    print_stability(b['x'], b['f'], b['s'], 6, 4, color=color, letter=True, linewidth=1.0,
                    ymin=y_in_c_min, ymax=y_in_c_max, xmin=xmin, xmax=xmax)
    plt.plot(network[:, 0], network[:, 5], 'x', color=color, ms=5.0, label='b=0')
    # plt.vlines(0.0, ymin=-30.0, ymax=200.0, color='m')
    plt.xticks(xticks)
    # plt.xlabel("external input", {"fontsize": labelticks_size})
    plt.yticks(y_in_c_ticks)
    # plt.ylabel("firing rate of excitatory\npopulation Hz", {"fontsize": labelticks_size})
    plt.tick_params(labelsize=ticks_size)
    plt.annotate('F', xy=(-0.15, 0.83), xycoords='axes fraction', weight='bold', fontsize=labelticks_size)

    # adaptation
    plt.sca(axs[2, 0])
    print_stability(b['x'], b['f'], b['s'], 6, 5, color=color, letter=True, linewidth=1.0,
                    ymin=y_ad_min, ymax=y_ad_max, xmin=xmin, xmax=xmax)
    # plt.plot(network_0[:, 0], network_0[:, 3], 'x', color=color, ms=5.0, label='b=0')
    # plt.vlines(0.0, ymin=-30.0, ymax=200.0, color='m')
    plt.xticks(xticks)
    plt.xlabel("external input (Hz)", {"fontsize": labelticks_size})
    plt.yticks(y_ad_ticks)
    plt.ylabel("ADAPTATION", {"fontsize": labelticks_size})
    plt.tick_params(labelsize=ticks_size)
    plt.annotate('G', xy=(-0.15, 0.83), xycoords='axes fraction', weight='bold', fontsize=labelticks_size)

    # no zoom adaptation
    fig.delaxes(axs[2, 1])

    # co-variance
    plt.sca(axs[2, 2])
    print_stability(b['x'], b['f'], b['s'], 6, 3, color=color, letter=True, linewidth=1.0,
                    ymin=y_ei_c_min, ymax=y_ei_c_max, xmin=xmin, xmax=xmax)
    plt.plot(network[:, 0], network[:, 4], 'x', color=color, ms=5.0, label='b=0')
    # plt.vlines(0.0, ymin=-30.0, ymax=200.0, color='m')
    plt.xticks(xticks)
    plt.xlabel("external input (Hz)", {"fontsize": labelticks_size})
    plt.yticks(y_ei_c_ticks)
    # plt.ylabel("firing rate of excitatory\npopulation Hz", {"fontsize": labelticks_size})
    plt.tick_params(labelsize=ticks_size)
    plt.annotate('H', xy=(-0.15, 0.83), xycoords='axes fraction', weight='bold', fontsize=labelticks_size)

    plt.subplots_adjust(top=0.960, bottom=0.095, left=0.16, right=0.965, wspace=0.26, hspace=0.215)
    plt.savefig('./figure/SP_figure_'+str(index)+'.png', dpi=300)
    # plt.show()
