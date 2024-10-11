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

## path of the data mean field
path = os.path.dirname(os.path.realpath(__file__)) + "/../../static/simulation/data/time_reduce_hist/"
# get firing rate
firing_rate_30 = np.load(path + '/b_30.0/firing_rate.npy')

letters = ['A', 'B']

## make figure
fig, axs = plt.subplots(2, 1, figsize=(6.8, 5.5))

for index, (firing_rate, b, color, title) in enumerate([(firing_rate_30, b_30, 'b', 'b=30 pA'), ]):
    plt.sca(axs[0])
    ## mean field
    print_stability(b['x'], b['f'], b['s'], 6, 0, color=color, letter=False, linewidth=linesize)
    ## plot high firing rate
    plt.plot(firing_rate[:61, 0], firing_rate[:61, 1], 'x', color='k', ms=size_marker_o, fillstyle='none')
    plt.plot(firing_rate[61:, 0], firing_rate[61:, 1], 'o', color='g', ms=size_marker_o, fillstyle='none')
    # plt.vlines(0.0, ymin=-30.0, ymax=200.0, color='m')
    plt.xlim(xmin=0.0, xmax=100.0)
    plt.xticks([0.0, 50.0, 100.0])
    # plt.xlabel("external input", {"fontsize": labelticks_size})
    plt.ylim(ymin=0.0, ymax=200.0)
    plt.yticks([0.0, 100.0, 200.0])
    plt.ylabel("firing rate of excitatory\npopulation (Hz)", {"fontsize": labelticks_size})
    # if index == 0:
    #     plt.ylabel("firing rate of excitatory\npopulation (Hz)", {"fontsize": labelticks_size})
    # plt.tick_params(labelsize=ticks_size)
    # plt.title(title)
    # plt.annotate(letters[index], xy=(-0.1, 0.9), xycoords='axes fraction', weight='bold', fontsize=labelticks_size)
    # for i in [0, 29, 59]:
    #     plt.plot(firing_rate[i, 0], firing_rate[i, 1], '.', color='black', ms=size_marker_o, fillstyle='full')
    #     if i != 59:
    #         plt.text(firing_rate[i, 0] - 1, firing_rate[i, 1] - 15, str(i + 1), color='black')
    #     else:
    #         plt.text(firing_rate[i, 0] - 3, firing_rate[i, 1] - 18, str(i + 1), color='black')
    # for i in [60, 89, 119]:
    #     plt.plot(firing_rate[i, 0], firing_rate[i, 1], '.', color='green', ms=size_marker_o, fillstyle='full')
    #     if i != 119:
    #         plt.text(firing_rate[i, 0]-1, firing_rate[i, 1]-15, str(i+1), color='green')
    #     else:
    #         plt.text(firing_rate[i, 0]-1, firing_rate[i, 1]+5, str(i+1), color='green')


    plt.sca(axs[1])
    ## mean field
    print_stability(b['x'], b['f'], b['s'], 6, 1, color=color, letter=False, linewidth=linesize)
    ## plot high firing rate
    # plt.plot(firing_rate[:, 0], firing_rate[:, 2], 'o', color=color, ms=size_marker_o, fillstyle='none')
    plt.plot(firing_rate[:61, 0], firing_rate[:61, 2], 'x', color='k', ms=size_marker_o, fillstyle='none')
    plt.plot(firing_rate[61:, 0], firing_rate[61:, 2], 'o', color='g', ms=size_marker_o, fillstyle='none')
    # plt.vlines(0.0, ymin=-30.0, ymax=200.0, color='m')
    plt.xlim(xmin=0.0, xmax=100.0)
    plt.xticks([0.0, 50.0, 100.0])
    plt.xlabel("external input (Hz)", {"fontsize": labelticks_size})
    plt.ylim(ymin=0.0, ymax=200.0)
    plt.yticks([0.0, 100.0, 200.0])
    plt.ylabel("firing rate of inhibitory\npopulation (Hz)", {"fontsize": labelticks_size})
    # if index == 0:
    #     plt.ylabel("firing rate of inhibitory\npopulation (Hz)", {"fontsize": labelticks_size})
    # plt.tick_params(labelsize=ticks_size)
    # plt.annotate(letters[index+1], xy=(-0.1, 0.9), xycoords='axes fraction', weight='bold', fontsize=labelticks_size)
    # for i in [0, 29, 59]:
    #     plt.plot(firing_rate[i, 0], firing_rate[i, 2], '.', color='black', ms=size_marker_o, fillstyle='full')
    #     if i != 59:
    #         plt.text(firing_rate[i, 0] - 1, firing_rate[i, 2] - 15, str(i + 1), color='black')
    #     else:
    #         plt.text(firing_rate[i, 0] - 3, firing_rate[i, 2] - 20, str(i + 1), color='black')
    # for i in [60, 89, 119]:
    #     plt.plot(firing_rate[i, 0], firing_rate[i, 2], '.', color='green', ms=size_marker_o, fillstyle='full')
    #     if i != 119:
    #         plt.text(firing_rate[i, 0]-1, firing_rate[i, 2]-15, str(i+1), color='green')
    #     else:
    #         plt.text(firing_rate[i, 0]-1, firing_rate[i, 2]+11, str(i+1), color='green')

plt.subplots_adjust(top=0.95, bottom=0.1, left=0.14, right=0.975, hspace=0.12, wspace=0.29)
plt.savefig('./figure/SP_figure_5_3_hist.png', dpi=300)
# plt.show()