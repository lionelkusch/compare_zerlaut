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
        if x < 50:
            s = "%.1f" % (range_values[int(x-1)] * 0.1)
        else:
            s = 'error index'
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
    violin_parts = axs.violinplot(data, widths=0.5, showmeans=True, showmedians=True, showextrema=True)
    for range_value, mean in zip(range(len(range_values)), data):
        axs.scatter(range_value+1, mean.mean(), marker='o', color='r', s=5.0)
    for pc in violin_parts['bodies']:
        pc.set_facecolor('b')
        pc.set_edgecolor('black')
    axs.set_xticks(np.arange(1, len(range_values))[::4])
    axs.tick_params(axis='x', labelrotation=90)
    axs.tick_params(labelsize=ticks_size)
    axs.xaxis.set_major_formatter(create_format(range_values))
    axs.yaxis.set_ticks([0., 0.5, 1.0, 1.5, 2.0])

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
lag = 100
# load result
path_init = os.path.dirname(os.path.realpath(__file__)) + "/../../static/simulation/data/long/"
values = gen_log_space(int((end - begin) / dt - window / dt) - int(window / dt) - nb_sample * dt - lag * 2,
                       nb_test) + int(lag * 2)
# get data
result_b_0_rate_10 = np.load(path_init + '/0.0_variance_timescale_10.0_1.npi.npy', allow_pickle=True)
result_b_60_rate_10 = np.load(path_init + '/60.0_variance_timescale_10.0_1.npi.npy', allow_pickle=True)

max_b_0_rate_10 = np.zeros_like(result_b_0_rate_10)
min_b_0_rate_10 = np.zeros_like(result_b_0_rate_10)
max_b_60_rate_10 = np.zeros_like(result_b_0_rate_10)
min_b_60_rate_10 = np.zeros_like(result_b_0_rate_10)
for max, min, result in [(max_b_0_rate_10, min_b_0_rate_10, result_b_0_rate_10),
                         (max_b_60_rate_10, min_b_60_rate_10, result_b_60_rate_10),]:
    for i in range(result.shape[1]):
        for index in range(result.shape[0]):
            data = np.array(result[index, i])
            data = data[np.where(data != -1)]
            max[index, i] = np.max(data)
            min[index, i] = np.max(data)
            result[index, i] = data[np.where(data != -1)]


fig, axs = plt.subplots(2, 2, figsize=(6.8, 5.5))
plot_violin(axs[0, 0], result_b_0_rate_10[0, :]/10, values, ticks_size)
axs[0, 0].set_ylabel('b:0 pA, input rate: 10 Hz\n timescale (ms)', {"fontsize": labelticks_size}, labelpad=0.)
axs[0, 0].annotate('A', xy=(-0.18, 0.95), xycoords='axes fraction', weight='bold', fontsize=labelticks_size)
axs[0, 0].set_title('excitatory population')
plot_violin(axs[0, 1], result_b_0_rate_10[1, :]/10, values, ticks_size)
axs[0, 1].set_title('inhibitory population')
axs[0, 1].annotate('B', xy=(-0.18, 0.95), xycoords='axes fraction', weight='bold', fontsize=labelticks_size)
plot_violin(axs[1, 0], result_b_60_rate_10[0, :]/10, values, ticks_size)
axs[1, 0].set_ylabel('b:60 pA, input rate: 10 Hz\n timescale (ms)', {"fontsize": labelticks_size}, labelpad=0.)
axs[1, 0].annotate('C', xy=(-0.18, 0.95), xycoords='axes fraction', weight='bold', fontsize=labelticks_size)
axs[1, 0].set_xlabel('length of time series (ms)', labelpad=0.0)
plot_violin(axs[1, 1], result_b_60_rate_10[1, :]/10, values, ticks_size)
axs[1, 1].annotate('D', xy=(-0.18, 0.95), xycoords='axes fraction', weight='bold', fontsize=labelticks_size)
axs[1, 1].set_xlabel('length of time series (ms)', labelpad=0.0)
plt.tick_params(labelsize=ticks_size)
plt.subplots_adjust(top=0.95, bottom=0.13, left=0.115, right=0.99, wspace=0.22, hspace=0.35)
plt.savefig('figure/SP_12_variance_timescale.png')
# plt.show()