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
linewidth = 1.0
marker_size = 3.0

## load bifurcation
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

## get network result
network_0 = np.concatenate([np.expand_dims(range(100), axis=1), np.load(path_init + '/0.0_mean_var.npy')], axis=1)
network_30 = np.concatenate([np.expand_dims(range(100), axis=1), np.load(path_init + '/30.0_mean_var.npy')], axis=1)
network_60 = np.concatenate([np.expand_dims(range(100), axis=1), np.load(path_init + '/60.0_mean_var.npy')], axis=1)

## make figure
fig, axs = plt.subplots(2, 2, figsize=(6.8, 5.5))

## excitatory population
plt.sca(axs[0, 0])
## plot stability of the mean field
line_b_0 = print_stability(b_0['x'], b_0['f'], b_0['s'], 6, 0, color='k', letter=False, linewidth=linewidth)
line_b_30 = print_stability(b_30['x'], b_30['f'], b_30['s'], 6, 0, color='b', letter=False, linewidth=linewidth)
line_b_60 = print_stability(b_60['x'], b_60['f'], b_60['s'], 6, 0, color='g', letter=False, linewidth=linewidth)
# plt.legend([line_b_60, line_b_30, line_b_0], ['b=60', 'b=30', 'b=0'], loc='lower right')
## plot network mean firing rate
plt.plot(network_0[:, 0], network_0[:, 1], 'xk', ms=marker_size, label='b=0')
plt.plot(network_30[:, 0], network_30[:, 1], 'xb', ms=marker_size, label='b=30')
plt.plot(network_60[:, 0], network_60[:, 1], 'xg', ms=marker_size, label='b=60')
## configure the figure
# plt.legend(loc='lower right')
# plt.vlines(0.0, ymin=-30.0, ymax=200.0, color='m')
plt.xlim(xmax=100.0, xmin=-0.1)
plt.xticks([0.0, 50.0, 100.0])
# plt.xlabel("external input", {"fontsize": labelticks_size})
plt.ylim(ymax=200.0, ymin=-0.1)
plt.yticks([0.0, 100.0, 200.0])
plt.ylabel("firing rate population (Hz)", {"fontsize": labelticks_size}, labelpad=0.0)
plt.tick_params(labelsize=ticks_size)
plt.title('excitatory population', {"fontsize": labelticks_size})
plt.annotate('A', xy=(-0.1, 1.05), xycoords='axes fraction', weight='bold', fontsize=labelticks_size)


## inhibitory population
plt.sca(axs[0, 1])
## plot stability of the mean field
line_b_0 = print_stability(b_0['x'], b_0['f'], b_0['s'], 6, 1, color='k', letter=False, linewidth=linewidth)
line_b_30 = print_stability(b_30['x'], b_30['f'], b_30['s'], 6, 1, color='b', letter=False, linewidth=linewidth)
line_b_60 = print_stability(b_60['x'], b_60['f'], b_60['s'], 6, 1, color='g', letter=False, linewidth=linewidth)
## plot network mean firing rate
plt.plot(network_0[:, 0], network_0[:, 3], 'xk', ms=marker_size, label='b=0')
plt.plot(network_30[:, 0], network_30[:, 3], 'xb', ms=marker_size, label='b=30')
plt.plot(network_60[:, 0], network_60[:, 3], 'xg', ms=marker_size, label='b=60')
## configure the figure
# plt.legend([line_b_60, line_b_30, line_b_0], ['b=60', 'b=30', 'b=0'], loc='lower right')
# plt.legend(loc='lower right')
# plt.vlines(0.0, ymin=-30.0, ymax=200.0, color='m')
plt.xlim(xmax=100.0, xmin=-0.1)
plt.xticks([0.0, 50.0, 100.0])
# plt.xlabel("external input", {"fontsize": labelticks_size})
plt.ylim(ymax=200.0, ymin=-0.1)
plt.yticks([0.0, 100.0, 200.0])
# plt.ylabel("firing rate of excitatory population Hz", {"fontsize": labelticks_size})
plt.tick_params(labelsize=ticks_size)
plt.title('inhibitory population', {"fontsize": labelticks_size})
plt.annotate('B', xy=(-0.1, 1.05), xycoords='axes fraction', weight='bold', fontsize=labelticks_size)


## excitatory population zoom
plt.sca(axs[1, 0])
## plot stability of the mean field
line_b_0 = print_stability(b_0['x'], b_0['f'], b_0['s'], 6, 0, color='k', letter=False, linewidth=linewidth)
line_b_30 = print_stability(b_30['x'], b_30['f'], b_30['s'], 6, 0, color='b', letter=False, linewidth=linewidth)
line_b_60 = print_stability(b_60['x'], b_60['f'], b_60['s'], 6, 0, color='g', letter=False, linewidth=linewidth)
## plot network mean firing rate
plt.plot(network_0[:, 0], network_0[:, 1], 'xk', ms=marker_size, label='b=0')
plt.plot(network_30[:, 0], network_30[:, 1], 'xb', ms=marker_size, label='b=30')
plt.plot(network_60[:, 0], network_60[:, 1], 'xg', ms=marker_size, label='b=60')
## configure the figure
# plt.legend([line_b_60, line_b_30, line_b_0], ['b=60', 'b=30', 'b=0'], loc='lower right')
# plt.legend(loc='lower right')
# plt.vlines(0.0, ymin=-30.0, ymax=200.0, color='m')
plt.xlim(xmax=50.0, xmin=-0.1)
plt.xticks([0.0, 25.0, 50.0])
plt.xlabel("external excitatory input (Hz)", {"fontsize": labelticks_size}, labelpad=2.)
plt.ylim(ymax=20.0, ymin=-0.1)
plt.yticks([0.0, 10.0, 20.0])
plt.ylabel("firing rate population (Hz)", {"fontsize": labelticks_size}, labelpad=6.)
plt.tick_params(labelsize=ticks_size)
plt.annotate('C', xy=(-0.1, 1.05), xycoords='axes fraction', weight='bold', fontsize=labelticks_size)


## inhibitory population zoom
plt.sca(axs[1, 1])
## plot stability of the mean field
line_b_0 = print_stability(b_0['x'], b_0['f'], b_0['s'], 6, 1, color='k', letter=False, linewidth=linewidth)
line_b_30 = print_stability(b_30['x'], b_30['f'], b_30['s'], 6, 1, color='b', letter=False, linewidth=linewidth)
line_b_60 = print_stability(b_60['x'], b_60['f'], b_60['s'], 6, 1, color='g', letter=False, linewidth=linewidth)
## plot network mean firing rate
plt.plot(network_0[:, 0], network_0[:, 3], 'xk', ms=marker_size, label='b=0')
plt.plot(network_30[:, 0], network_30[:, 3], 'xb', ms=marker_size, label='b=30')
plt.plot(network_60[:, 0], network_60[:, 3], 'xg', ms=marker_size, label='b=60')
## configure the figure
# plt.legend([line_b_60, line_b_30, line_b_0], ['b=60', 'b=30', 'b=0'], loc='lower right')
# plt.legend(loc='lower right')
# plt.vlines(0.0, ymin=-30.0, ymax=200.0, color='m')
plt.xlim(xmax=50.0, xmin=-0.1)
plt.xticks([0.0, 25.0, 50.0])
plt.xlabel("external excitatory input (Hz)", {"fontsize": labelticks_size}, labelpad=2.)
plt.ylim(ymax=120.0, ymin=-0.1)
plt.yticks([0.0, 60.0, 120.0])
# plt.ylabel("firing rate of excitatory population Hz", {"fontsize": labelticks_size})
plt.tick_params(labelsize=ticks_size)
plt.annotate('D', xy=(-0.1, 1.05), xycoords='axes fraction', weight='bold', fontsize=labelticks_size)

plt.subplots_adjust(top=0.95, bottom=0.10, left=0.09, right=0.975, wspace=0.16, hspace=0.15)

# plt.show()
plt.savefig('./figure/figure_1.png', dpi=300)
