#  Copyright 2023 Aix-Marseille Université
# "Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements; and to You under the Apache License, Version 2.0. "
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from parameter_analyse.static.python_file.plot.print_study_time_std import gen_log_space


def create_format(range_values):
    """
    format for the ticks
    :param range_values:
    :return:
    """
    @ticker.FuncFormatter
    def format_window(x, pos):
        s = "%.1f" % (range_values[int(x-1)] * 0.1)
        return s
    return format_window


def plot_violin(axs, data, range_values, ticks_size):
    """
    violin plot
    :param axs: axis to plot
    :param data: data
    :param range_values: value
    :param ticks_size: size of ticks
    :return:
    """
    # violin_parts = axs.violinplot(data, positions=range_values, widths=0.5, showmeans=True, showmedians=True)
    violin_parts = axs.violinplot(data, widths=0.5, showmeans=True, showmedians=True)
    for range_value, mean in zip(range(len(range_values)), data):
        # axs.scatter(np.repeat(range_value + .8, len(mean)), mean, color='black', s=0.1)
        axs.scatter(range_value+1, mean.mean(), marker='o', color='r', s=5.0)
    for pc in violin_parts['bodies']:
        pc.set_facecolor('b')
        pc.set_edgecolor('black')
    # axs.hlines(data[-1][0], xmin=1, xmax=len(range_values), color='r', alpha=0.5)
    axs.set_xticks(np.arange(1, len(range_values) + 1)[::4])
    axs.set_xlabel('ms', labelpad=0.0)
    axs.tick_params(axis='x', labelrotation=90)
    axs.tick_params(labelsize=ticks_size)
    axs.xaxis.set_major_formatter(create_format(range_values))

# parameters for getting data
dt = 0.1
window = 5.0
begin = 1000.0
end = 5000.0
nb_test = 50
nb_sample = 50000
labelticks_size = 12
label_legend_size = 12
ticks_size = 10
# add different window data
for b, rate in [(0.0, 10.0), (0.0, 50.0), (0.0, 60.0),
                (30.0, 10.0), (30.0, 50.0), (30.0, 60.0),
                (60.0, 10.0), (60.0, 50.0), (60.0, 60.0)]:
    path_init = os.path.dirname(os.path.realpath(__file__)) + "/../../static/simulation/data/long/"
    values = gen_log_space(int((end - begin) / dt - window / dt) - int(window / dt) - nb_sample * dt, nb_test)[2:]
    # get data
    result = np.load(path_init + '/' + str(b) + '_size_variance_rate_' + str(rate) + '.npy', allow_pickle=True)
    mean_ex = []
    mean_in = []
    for mean in np.concatenate([result[0, 2:]]):
        if len(mean) >= 50000:
            mean_ex.append(np.array(mean, dtype=float)[:, 0])
            mean_in.append(np.array(mean, dtype=float)[:, 1])
    variances_ex = []
    variances_in = []
    covariance = []
    for cov in np.concatenate([result[1, 2:]]):
        if len(cov) >= 50000:
            variances_ex.append(np.array(cov, dtype=float)[:, 0, 0])
            variances_in.append(np.array(cov, dtype=float)[:, 1, 1])
            covariance.append(np.array(cov, dtype=float)[:, 1, 0])

    fig, axs = plt.subplots(2, 3, figsize=(6.8, 5.5))
    plot_violin(axs[0, 0], mean_ex, values[:len(mean_ex)], ticks_size)
    axs[0, 0].set_ylabel('firing rate (Hz)', {"fontsize": labelticks_size}, labelpad=0.)
    axs[0, 0].annotate('A', xy=(-0.23, 0.95), xycoords='axes fraction', weight='bold', fontsize=labelticks_size)
    plot_violin(axs[0, 1], mean_in, values[:len(mean_ex)], ticks_size)
    axs[0, 1].annotate('B', xy=(-0.23, 0.95), xycoords='axes fraction', weight='bold', fontsize=labelticks_size)
    plot_violin(axs[1, 0], variances_ex, values[:len(mean_ex)], ticks_size)
    axs[1, 0].set_ylabel('variance', {"fontsize": labelticks_size}, labelpad=6.)
    axs[1, 0].annotate('C', xy=(-0.10, 0.88), xycoords='axes fraction', weight='bold', fontsize=labelticks_size)
    plot_violin(axs[1, 1], variances_in, values[:len(mean_ex)], ticks_size)
    axs[1, 1].annotate('D', xy=(-0.10, 0.88), xycoords='axes fraction', weight='bold', fontsize=labelticks_size)
    plot_violin(axs[1, 2], covariance, values[:len(mean_ex)], ticks_size)
    axs[1, 2].annotate('E', xy=(-0.10, 0.88), xycoords='axes fraction', weight='bold', fontsize=labelticks_size)
    fig.delaxes(axs[0, 2])
    plt.tick_params(labelsize=ticks_size)
    plt.subplots_adjust(top=0.98, bottom=0.13, left=0.08, right=0.98, wspace=0.25, hspace=0.30)
    plt.savefig('figure/SP_figure_4_'+str(b) + '_rate_' + str(rate) + '.pdf')
    plt.close('all')
    # plt.show()