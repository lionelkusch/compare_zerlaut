#  Copyright 2023 Aix-Marseille Université
# "Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements; and to You under the Apache License, Version 2.0. "
import os
import matplotlib.pyplot as plt
import numpy as np
from parameter_analyse.static.python_file.plot.helper_function import get_gids_all, load_spike_all_long, \
    detection_burst


def get_firing_rate(path_init, firing_rate_ext_init=52.0, firing_rate_end=-0.1,
                    increment_firing_rate=-1.0, interval_time=10000.0):
    """
    save in a file the firing rate of each step
    :param path_init: path of the files
    :param firing_rate_ext_init: initial firing rate
    :param firing_rate_end: minimal firing rate
    :param increment_firing_rate: negative increment of the firing rate
    :param interval_time: interval of time for each step
    :return:
    """
    gids_all = get_gids_all(path_init)
    nb_ex = gids_all['excitatory'][0][1] - gids_all['excitatory'][0][0] + 1
    nb_in = gids_all['inhibitory'][0][1] - gids_all['inhibitory'][0][0] + 1
    data_pop_all = load_spike_all_long(path_init, firing_rate_ext_init, firing_rate_end, increment_firing_rate)
    firing_rates = []
    for firing_rate in np.arange(firing_rate_ext_init, firing_rate_end, increment_firing_rate):
        name_firing_rate = str(np.around(firing_rate))
        firing_rates.append([data_pop_all[name_firing_rate]['excitatory'].shape[0] / nb_ex / (interval_time * 1e-3),
                             data_pop_all[name_firing_rate]['inhibitory'].shape[0] / nb_in / (interval_time * 1e-3)])
    np.save(path_init + '/firing_rate.npy', firing_rates)


def plot_firing_rate(path_init):
    """
    plot firing rate of the particular simulation
    :param path_init: path of the result
    :return:
    """
    firing_rate = np.load(path_init + '/firing_rate.npy')
    plt.plot(firing_rate[:, 0][:-1], 'x')
    plt.plot(firing_rate[:, 1][:-1], 'x')
    plt.show()


def plot_firing_rate_all(path_inits):
    """
    plotting the firing rate for multiple simulation
    :param path_inits: array of path of the result
    :return:
    """
    for path_init in path_inits:
        firing_rate = np.load(path_init + '/firing_rate.npy')
        plt.plot(firing_rate[:, 0][:-1], 'x')
        plt.plot(firing_rate[:, 1][:-1], 'x')
    plt.show()


def plot_spiketrains(path_init, firing_rate_ext, font_size=10.0, tickfont_size=7.0, burst=False, size_mark=0.1):
    """
    plot spike trains of particular step
    :param path_init: path of the data
    :param firing_rate_ext: value of the external firing rate
    :param font_size: font size of the label
    :param tickfont_size: fontsize of the ticks
    :param burst: add line with burst
    :param size_mark: size of the mark
    :return:
    """
    gids_all = get_gids_all(path_init)
    name_firing_rate = str(np.around(firing_rate_ext))
    data_pop_all = load_spike_all_long(path_init, firing_rate_ext, firing_rate_ext - 1, -1)
    for key, value in zip(data_pop_all[name_firing_rate].keys(), data_pop_all[name_firing_rate].values()):
        data_pop_all[name_firing_rate][key] = np.swapaxes(value, 0, 1)
    begin = np.min([np.min(data_pop_all[name_firing_rate][name][1, :]) for (name, gids) in gids_all.items()])
    end = np.max([np.max(data_pop_all[name_firing_rate][name][1, :]) for (name, gids) in gids_all.items()])
    color = ['red', 'blue']
    for pop, [neurons_id, times_spike] in enumerate(data_pop_all[name_firing_rate].values()):
        plt.plot(times_spike, neurons_id, '.', color=color[pop], markersize=size_mark)
    if burst:
        for pop, (name, gids) in enumerate(gids_all.items()):
            burst = detection_burst(gids, data_pop_all[name_firing_rate][name], begin, end, limit_burst=10.0)
            if len(burst) != 0:
                plt.plot([burst[1], burst[2]], [burst[0], burst[0]], color=color[pop], linewidth=1.0, alpha=0.2)
            else:
                print("No Burst for " + name)
    plt.tick_params(axis='both', labelsize=tickfont_size)
    plt.xlim(xmax=end + 10.0, xmin=begin - 10.0)
    plt.xlabel('time in ms', {"fontsize": font_size}, labelpad=2.5)
    plt.show()


if __name__ == '__main__':
    # high fixed point continue estimation
    path_init = os.path.dirname(os.path.realpath(__file__)) + "/../../simulation/data/time_reduce/"
    firing_rate_ext_init = 52.0
    firing_rate_min = -0.1
    # plot_spiketrains(path_init + '/b_0.0/', firing_rate_ext=0.0, burst=False)
    get_firing_rate(path_init+'/b_0.0/')
    # plot_firing_rate(path_init+'/b_0.0/')
    get_firing_rate(path_init+'/b_30.0/')
    # plot_firing_rate(path_init+'/b_30.0/')
    get_firing_rate(path_init+'/b_60.0/')
    # plot_firing_rate(path_init+'/b_60.0/')
    # plot_firing_rate_all([path_init + '/b_0.0/', path_init + '/b_30.0/', path_init + '/b_60.0/'])

    # low fixed point continue estimation
    path_init = os.path.dirname(os.path.realpath(__file__)) + "/../../simulation/data/time_reduce_low/"
    firing_rate_ext_init = 48.0
    firing_rate_min = -0.1
    # get_firing_rate(path_init+'/b_0.0/', firing_rate_ext_init=48.0, firing_rate_end=99.0, increment_firing_rate=1.0)
    plot_firing_rate(path_init+'/b_0.0/')
    # get_firing_rate(path_init+'/b_30.0/', firing_rate_ext_init=48.0, firing_rate_end=99.0, increment_firing_rate=1.0)
    plot_firing_rate(path_init+'/b_30.0/')
    # get_firing_rate(path_init+'/b_60.0/', firing_rate_ext_init=48.0, firing_rate_end=99.0, increment_firing_rate=1.0)
    plot_firing_rate(path_init+'/b_60.0/')