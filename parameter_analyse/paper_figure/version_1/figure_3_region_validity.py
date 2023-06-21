#  Copyright 2023 Aix-Marseille Université
# "Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements; and to You under the Apache License, Version 2.0. "
import os
import numpy as np
import matplotlib.pyplot as plt
from parameter_analyse.spike_oscilation.python_file.print.print_exploration_analysis import getData, grid, draw_point, draw_line_level


def draw_contour(fig, ax, X, Y, Z, levels, title, xlabel, ylabel, zlabel, zmin, zmax, label_size,
                       number_size, color_bar=False):
    """
    create graph with contour with limit
    :param fig: figure where to plot
    :param ax: axis of the graph
    :param X: x values
    :param Y: y values
    :param Z: z values
    :param levels: levels of network
    :param title: title of the graph
    :param xlabel: x label
    :param ylabel: y label
    :param zlabel: z label
    :param zmin: z minimum values
    :param zmax: z maximum values
    :param label_size: size of the label
    :param number_size: size of the number
    :param color_bar: add or not color bar
    :return:
    """
    CS = ax.tricontourf(X, Y, Z, False, levels=levels, vmin=zmin, vmax=zmax, extend='both')
    if color_bar:
        cbar = fig.colorbar(CS, ax=ax)
        cbar.ax.set_ylabel(zlabel, {"fontsize": label_size})
        cbar.ax.tick_params(labelsize=number_size)
        cbar.ax.set_ylabel('ms', {"fontsize": label_legend_size}, labelpad=0.)
    ax.set_xlabel(xlabel, {"fontsize": label_size})
    ax.set_ylabel(ylabel, {"fontsize": label_size})
    ax.set_title(title, {"fontsize": label_size})

## parameter of the figures
labelticks_size = 12
label_legend_size = 12
ticks_size = 12
size_point = 1.0
## path of the data
path = os.path.dirname(__file__) + '/../../spike_oscilation/simulation/'
data_base_network_0 = path + '/simulation/rate_0.0/amplitude_frequency.db'
data_base_network_7 = path + '/simulation/rate_7.0/amplitude_frequency.db'
data_base_network_amplitude = path + '/simulation/rate_amplitude/amplitude_frequency.db'
data_base_network_adp_0 = path + '/simulation_b_60/rate_0.0/amplitude_frequency.db'
data_base_network_adp_7 = path + '/simulation_b_60/rate_7.0/amplitude_frequency.db'
data_base_network_adp_amplitude = path + '/simulation_b_60/rate_amplitude/amplitude_frequency.db'
## information for databased
table_name_network = 'first_exploration'
population = 'excitatory'
list_variable = [{'name': 'amplitude', 'title': 'amplitude ', 'min': 0.0, 'max': 5000000.0},
                 {'name': 'frequency', 'title': 'frequency input', 'min': 0.0, 'max': 5000000.0}]
letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G']

## make figure
fig, axs = plt.subplots(2, 3, gridspec_kw={'width_ratios': [0.9, 0.9, 1.2]}, figsize=(6.8, 5.5))
for index, data_base in enumerate([data_base_network_amplitude, data_base_network_0, data_base_network_7,
                                   data_base_network_adp_amplitude, data_base_network_adp_0, data_base_network_adp_7]):
    ax = axs[index // 3, index % 3]
    data_network = getData(data_base, table_name_network, list_variable, population)
    id = np.where(np.logical_and(data_network['timescale_w5ms'], data_network['ISI_min']))
    X, Y, Z = grid(data_network['amplitude'], data_network['frequency'], data_network['timescale_w5ms'], res=False,
                   resX=None, resY=None, id=id)
    X1, Y1, Z1 = grid(data_network['amplitude'], data_network['frequency'], data_network['ISI_min'], res=False,
                      resX=None, resY=None, id=id)
    draw_contour(fig, ax, X, Y, Z, [0.0, 6.0, 15.0, 30.0, 45.0, 70.0],
                 '', '', '', '', 0.0, 70.0, 10.0, 11.0, color_bar=(index % 3 == 2))
    draw_point(ax, X, Y, size=size_point)
    draw_line_level(ax, X, Y, Z - Z1, False, 0.0, 'red')

    if index == 0:
        ## timescale amplitude
        ax.set_title('mean rate : amplitude', {"fontsize": labelticks_size})
        ax.set_ylabel('frequency of input (Hz)', {"fontsize": labelticks_size}, labelpad=2.)
        ax.tick_params(labelsize=ticks_size)
    elif index == 1:
        ## timescale 0.0
        ax.set_title('mean rate : 0.0 Hz', {"fontsize": labelticks_size})
        ax.tick_params(labelsize=ticks_size)
    elif index == 2:
        ## timescale 7.0
        ax.set_title('mean rate : 7.0 Hz', {"fontsize": labelticks_size})
        ax.tick_params(labelsize=ticks_size)
    elif index == 3:
        ## timescale with adaptation
        ax.set_ylabel('frequency of input (Hz)', {"fontsize": labelticks_size}, labelpad=2.)
        ax.set_xlabel('amplitude of input(Hz)', {"fontsize": labelticks_size}, labelpad=0.)
        ax.tick_params(labelsize=ticks_size)
    elif index == 4:
        ## timescale with adaptation 0.0
        ax.set_xlabel('amplitude of input(Hz)', {"fontsize": labelticks_size}, labelpad=0.)
        ax.tick_params(labelsize=ticks_size)
    elif index == 5:
        ## timescale with adaptation 7.0
        ax.set_xlabel('amplitude of input(Hz)', {"fontsize": labelticks_size}, labelpad=0.)
        ax.tick_params(labelsize=ticks_size)
    ax.annotate(letters[index], xy=(-0.15, 0.9), xycoords='axes fraction', weight='bold', fontsize=labelticks_size)


plt.subplots_adjust(top=0.96, bottom=0.085, left=0.08, right=0.965, wspace=0.19, hspace=0.105)
# plt.show()
plt.savefig('./figure/figure_3.png', dpi=300)
