#  Copyright 2023 Aix-Marseille Université
# "Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements; and to You under the Apache License, Version 2.0. "
import matplotlib.pyplot as plt
import os
from parameter_analyse.static.python_file.plot.helper_function import get_gids_all, load_spike_all, detection_burst


def plot_spiketrains(path_init, ax, font_size=10.0, tickfont_size=7.0, burst=False, size_mark=0.1, begin=1000.0,
                     end=5000.0):
    """
    plot spike trains
    :param path_init: path of the result simulation
    :param ax: axis of the figure for plotting
    :param font_size: size of label font
    :param tickfont_size: size of the ticks size
    :param burst: add line when their are a burst
    :param size_mark: size of the mark
    :param begin: start of the plot
    :param end: end of the plot
    :return:
    """
    gids_all = get_gids_all(path_init)
    data_pop_all = load_spike_all(gids_all, path_init, begin, end)
    color = ['red', 'blue']
    for pop, [neurons_id, times_spike] in enumerate(data_pop_all.values()):
        ax.plot(times_spike, neurons_id, '.', color=color[pop], markersize=size_mark)
    if burst:
        for pop, (name, gids) in enumerate(gids_all.items()):
            burst = detection_burst(gids, data_pop_all[name], begin, end, limit_burst=10.0)
            if len(burst) != 0:
                ax.plot([burst[1], burst[2]], [burst[0], burst[0]], color=color[pop], linewidth=1.0, alpha=0.2)
            else:
                print("No Burst for " + name)
    ax.tick_params(axis='both', labelsize=tickfont_size)
    ax.set_xlim(xmax=end + 10.0, xmin=begin - 10.0)
    ax.set_xlabel('time in ms', {"fontsize": font_size}, labelpad=2.5)


if __name__ == '__main__':
    path_init = os.path.dirname(os.path.realpath(__file__)) + "/../../simulation/data/"
    path = path_init + '/master_seed_0/_b_60.0_rate_52.0/'; begin = 1000.0; end = 5000.0
    # path = path_init+'/time_reduce/b_60.0/'; begin=8000.0; end=23000.0
    plt.figure(figsize=(20, 10))
    ax = plt.gca()
    plot_spiketrains(path, ax, font_size=30.0, tickfont_size=20.0, burst=False, size_mark=0.1, begin=begin, end=end)
    # plt.subplots_adjust(top=0.99, bottom=0.11, left=0.05, right=0.98, hspace=0.2, wspace=0.2)
    plt.show()

    plt.figure(figsize=(20, 10))
    ax = plt.gca()
    path = path_init + '/short/_b_0.0_rate_10.0/'; begin = 0.0; end = 2000.0
    plot_spiketrains(path, ax, font_size=30.0, tickfont_size=20.0, burst=False, size_mark=0.1, begin=begin, end=end)
    plt.figure(figsize=(20, 10))
    ax = plt.gca()
    path = path_init + '/short/_b_0.0_rate_60.0/'; begin = 0.0; end = 2000.0
    plot_spiketrains(path, ax, font_size=30.0, tickfont_size=20.0, burst=False, size_mark=0.1, begin=begin, end=end)
    plt.show()
