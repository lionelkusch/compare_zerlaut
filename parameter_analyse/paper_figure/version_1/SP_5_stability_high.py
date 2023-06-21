#  Copyright 2023 Aix-Marseille Université
# "Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements; and to You under the Apache License, Version 2.0. "
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.matlab import loadmat
from parameter_analyse.analyse_dynamic.print_figure.print_stability import print_stability

## parameter of the figures
labelticks_size = 12
label_legend_size = 12
ticks_size = 12
size_marker = 3.0
size_marker_o = 3.0
linesize = 1.0

## path of the data network
path = os.path.dirname(__file__) + '/../../analyse_dynamic/matlab/'
path_init = os.path.dirname(os.path.realpath(__file__)) + "/../../static/simulation/data/master_seed_0/"
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

# get network result
network_0 = np.concatenate([np.expand_dims(range(100), axis=1), np.load(path_init + '/0.0_mean_var.npy')], axis=1)
network_30 = np.concatenate([np.expand_dims(range(100), axis=1), np.load(path_init + '/30.0_mean_var.npy')], axis=1)
network_60 = np.concatenate([np.expand_dims(range(100), axis=1), np.load(path_init + '/60.0_mean_var.npy')], axis=1)

## path of the data mean field
path = os.path.dirname(os.path.realpath(__file__)) + "/../../static/simulation/data/time_reduce/"
# get firing rate
firing_rate_0 = np.load(path + '/b_0.0/firing_rate.npy')
firing_rate_0 = np.concatenate((np.array([np.arange(52.0, -0.1, -1.0)]).swapaxes(0, 1), firing_rate_0), axis=1)
firing_rate_30 = np.load(path + '/b_30.0/firing_rate.npy')
firing_rate_30 = np.concatenate((np.array([np.arange(52.0, -0.1, -1.0)]).swapaxes(0, 1), firing_rate_30), axis=1)
firing_rate_60 = np.load(path + '/b_60.0/firing_rate.npy')
firing_rate_60 = np.concatenate((np.array([np.arange(52.0, -0.1, -1.0)]).swapaxes(0, 1), firing_rate_60), axis=1)

letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G']

## make figure
fig, axs = plt.subplots(2, 3, figsize=(6.8, 5.5))

for index, (network, firing_rate, b, color, title) in enumerate([(network_0,  firing_rate_0,  b_0,  'k', 'b=0 pA'),
                                                                 (network_30, firing_rate_30, b_30, 'b', 'b=30 pA'),
                                                                 (network_60, firing_rate_60, b_60, 'g', 'b=60 pA')]):
    plt.sca(axs[0, index])
    ## mean field
    print_stability(b['x'], b['f'], b['s'], 6, 0, color=color, letter=False, linewidth=linesize)
    ## plot high firing rate
    plt.plot(firing_rate[:, 0], firing_rate[:, 1], 'o', color=color, ms=size_marker_o, fillstyle='none')
    ## plot network rate
    plt.plot(network[:, 0], network[:, 1], 'x', color=color, ms=size_marker)
    # plt.vlines(0.0, ymin=-30.0, ymax=200.0, color='m')
    plt.xlim(xmin=-1.0, xmax=52.0)
    plt.xticks([0.0, 25.0, 50.0])
    # plt.xlabel("external input", {"fontsize": labelticks_size})
    plt.ylim(ymin=0.0, ymax=200.0)
    plt.yticks([0.0, 100.0, 200.0])
    if index == 0:
        plt.ylabel("firing rate of excitatory\npopulation (Hz)", {"fontsize": labelticks_size})
    plt.tick_params(labelsize=ticks_size)
    plt.title(title)
    plt.annotate(letters[index], xy=(-0.1, 0.9), xycoords='axes fraction', weight='bold', fontsize=labelticks_size)

    plt.sca(axs[1, index])
    ## mean field
    print_stability(b['x'], b['f'], b['s'], 6, 1, color=color, letter=False, linewidth=linesize)
    ## plot high firing rate
    plt.plot(firing_rate[:, 0], firing_rate[:, 2], 'o', color=color, ms=size_marker_o, fillstyle='none')
    ## plot network rate
    plt.plot(network[:, 0], network[:, 3], 'x', color=color, ms=size_marker)
    # plt.vlines(0.0, ymin=-30.0, ymax=200.0, color='m')
    plt.xlim(xmin=-1.0, xmax=52.0)
    plt.xticks([0.0, 25.0, 50.0])
    plt.xlabel("external input (Hz)", {"fontsize": labelticks_size})
    plt.ylim(ymin=0.0, ymax=200.0)
    plt.yticks([0.0, 100.0, 200.0])
    if index == 0:
        plt.ylabel("firing rate of inhibitory\npopulation (Hz)", {"fontsize": labelticks_size})
    plt.tick_params(labelsize=ticks_size)
    plt.annotate(letters[index+3], xy=(-0.1, 0.9), xycoords='axes fraction', weight='bold', fontsize=labelticks_size)

plt.subplots_adjust(top=0.95, bottom=0.1, left=0.14, right=0.99, hspace=0.12, wspace=0.29)
plt.savefig('./figure/SP_figure_5.png', dpi=300)
# plt.show()