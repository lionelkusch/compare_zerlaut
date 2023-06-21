import os
import matplotlib.pyplot as plt
from parameter_analyse.zerlaut_oscilation.python_file.print.print_result import getData, grid, draw_point


def draw_contour(fig, ax, X, Y, Z, levels, title, xlabel, ylabel, zlabel, zmin, zmax, label_size,
                 number_size):
    """
    create graph with contour with limit
    :param fig: figure where to plot
    :param ax: axis of the graph
    :param X: x values
    :param Y: y values
    :param Z: z values
    :param levels: levels for color
    :param title: title of the graph
    :param xlabel: x label
    :param ylabel: y label
    :param zlabel: z label
    :param zmin: z minimum values
    :param zmax: z maximum values
    :param label_size: size of the label
    :param number_size: size of the number
    :return:
    """
    CS = ax.tricontourf(X, Y, Z, False, levels=levels, vmin=zmin, vmax=zmax, extend='both')
    cbar = fig.colorbar(CS, ax=ax)
    cbar.ax.set_ylabel(zlabel, {"fontsize": label_size})
    cbar.ax.tick_params(labelsize=number_size)
    ax.set_xlabel(xlabel, {"fontsize": label_size})
    ax.set_ylabel(ylabel, {"fontsize": label_size})
    ax.set_title(title, {"fontsize": label_size})
    return cbar


## parameter of the figures
labelticks_size = 12
label_legend_size = 12
ticks_size = 12
## path of the data
path_network = os.path.dirname(__file__) + '/../../spike_oscilation/simulation/'
data_base_network_0 = path_network + '/simulation/rate_0.0/amplitude_frequency.db'
data_base_network_7 = path_network + '/simulation/rate_7.0/amplitude_frequency.db'
data_base_network_amplitude = path_network + '/simulation/rate_amplitude/amplitude_frequency.db'
data_base_network_adp_0 = path_network + '/simulation_b_60/rate_0.0/amplitude_frequency.db'
data_base_network_adp_7 = path_network + '/simulation_b_60/rate_7.0/amplitude_frequency.db'
data_base_network_adp_amplitude = path_network + '/simulation_b_60/rate_amplitude/amplitude_frequency.db'
path_mean = os.path.dirname(__file__) + '/../../zerlaut_oscilation/simulation/'
data_base_mean = path_mean + '/deterministe/database.db'
## information for databased
table_name_network = 'first_exploration'
table_name_mean = 'exploration'
population = 'excitatory'
list_variable = [{'name': 'amplitude', 'title': 'amplitude ', 'min': 0.0, 'max': 5000000.0},
                 {'name': 'frequency', 'title': 'frequency input', 'min': 0.0, 'max': 5000000.0}]
# choose the rate
for rate, name_figure, adp in [
                               (7.0, './figure/figure_4.png', False), (0.0, './figure/SP_figure_9.png', False),
                               (7.0, './figure/SP_figure_9_7_adpt.png', True), (0.0, './figure/SP_figure_9_0_adpt.png', True),
                               # (-1.0, './figure/SP_figure_9_amp.png', False), (-1.0, './figure/SP_figure_9_amp_1.png', True),
                               ]:
    if rate == 0.0:
        data_base_network = data_base_network_0 if not adp else data_base_network_adp_0
    elif rate == 7.0:
        data_base_network = data_base_network_7 if not adp else data_base_network_adp_7
    elif rate == -1.0:
        data_base_network = data_base_network_amplitude if not adp else data_base_network_adp_amplitude
    else:
        raise Exception('bad rate')

    ## make figure
    fig = plt.figure(figsize=(6.8, 5.5))

    #### network
    ax = plt.subplot(2, 3, 1)
    data_network = getData(data_base_network, table_name_network, list_variable, population)
    X, Y, Z = grid(data_network['amplitude'], data_network['frequency'], data_network['PLV_w5ms'], res=False,
                   resX=None, resY=None)
    draw_contour(fig, ax, X, Y, Z, [0., 0.5, 0.6, 0.7, 0.8, 0.9, 1.], '', '', '', '', 0.0, 1.0, 10.0, ticks_size)
    draw_point(ax, X, Y, size=2.0)
    ax.set_ylabel('Network\nfrequency of input (Hz)', fontdict={'fontsize': labelticks_size})
    ax.set_title('Phase Locking Value', fontdict={'fontsize': labelticks_size})
    ax.tick_params(labelsize=ticks_size)
    ax.annotate('A', xy=(-0.15, 0.90), xycoords='axes fraction', weight='bold', fontsize=labelticks_size)

    #### mean field
    ax = plt.subplot(2, 3, 4)
    data_network = getData(data_base_mean, table_name_mean, list_variable, population, cond=" AND rate == " + str(rate))
    X, Y, Z = grid(data_network['amplitude'], data_network['frequency'], data_network['PLV_value'], res=False,
                   resX=None, resY=None)
    draw_contour(fig, ax, X, Y, Z, [0., 0.5, 0.6, 0.7, 0.8, 0.9, 1.], '', '', '', '', 0.0, 1.0, 10.0, ticks_size)
    draw_point(ax, X, Y, size=2.0)
    ax.set_ylabel('Mean field\nfrequency of input (Hz)', fontdict={'fontsize': labelticks_size})
    ax.set_xlabel('amplitude of input(Hz)', {"fontsize": labelticks_size}, labelpad=0.)
    ax.tick_params(labelsize=ticks_size)
    ax.annotate('D', xy=(-0.15, 0.90), xycoords='axes fraction', weight='bold', fontsize=labelticks_size)

    #### network
    ax = plt.subplot(2, 3, 2)
    data_network = getData(data_base_network, table_name_network, list_variable, population)
    X, Y, Z = grid(data_network['amplitude'], data_network['frequency'], data_network['MeanPhaseShift_5ms'], res=False,
                   resX=None, resY=None)
    draw_contour(fig, ax, X, Y, Z, [-1., -0.75, -0.5, -0.25, 0., 0.25, 0.5, 0.75, 1.], '', '', '', '', -1.0, 1.0, 10.0,
                 ticks_size)
    draw_point(ax, X, Y, size=2.0)
    ax.set_title('Phase shift (rad)', fontdict={'fontsize': labelticks_size})
    ax.tick_params(labelsize=ticks_size)
    ax.set_yticks([])
    ax.annotate('B', xy=(-0.15, 0.90), xycoords='axes fraction', weight='bold', fontsize=labelticks_size)

    #### mean field
    ax = plt.subplot(2, 3, 5)
    data_network = getData(data_base_mean, table_name_mean, list_variable, population, cond=" AND rate == " + str(rate))
    X, Y, Z = grid(data_network['amplitude'], data_network['frequency'], data_network['PLV_angle'], res=False, resX=None,
                   resY=None)
    draw_contour(fig, ax, X, Y, Z, [-1., -0.75, -0.5, -0.25, 0., 0.25, 0.5, 0.75, 1.], '', '', '', '', -1.0, 1.0, 10.0,
                 ticks_size)
    draw_point(ax, X, Y, size=2.0)
    ax.set_xlabel('amplitude of input(Hz)', {"fontsize": labelticks_size}, labelpad=0.)
    ax.tick_params(labelsize=ticks_size)
    ax.set_yticks([])
    ax.annotate('E', xy=(-0.15, 0.90), xycoords='axes fraction', weight='bold', fontsize=labelticks_size)

    #### network
    ax = plt.subplot(2, 3, 3)
    data_network = getData(data_base_network, table_name_network, list_variable, population)
    X, Y, Z = grid(data_network['amplitude'], data_network['frequency'], data_network['max_IFR_w5ms'], res=False,
                   resX=None, resY=None)
    draw_contour(fig, ax, X, Y, Z, [0., 5.0, 10.0, 15.0, 20.0], '', '', '', '', 0.0, 20.0, 10.0, ticks_size)
    draw_point(ax, X, Y, size=2.0)
    ax.set_title('Maximum firing rate (Hz)', fontdict={'fontsize': labelticks_size})
    ax.tick_params(labelsize=ticks_size)
    ax.set_yticks([])
    ax.annotate('C', xy=(-0.15, 0.90), xycoords='axes fraction', weight='bold', fontsize=labelticks_size)

    #### mean field
    ax = plt.subplot(2, 3, 6)
    data_network = getData(data_base_mean, table_name_mean, list_variable, population, cond=" AND rate == " + str(rate))
    X, Y, Z = grid(data_network['amplitude'], data_network['frequency'], data_network['max_rates'], res=False,
                   resX=None, resY=None)
    draw_contour(fig, ax, X, Y, Z, [0., 5.0, 10.0, 15.0, 20.0], '', '', '', '', 0.0, 20.0, 10.0, ticks_size)
    draw_point(ax, X, Y, size=2.0)
    ax.set_xlabel('amplitude of input(Hz)', {"fontsize": labelticks_size}, labelpad=0.)
    ax.tick_params(labelsize=ticks_size)
    ax.set_yticks([])
    ax.annotate('F', xy=(-0.15, 0.90), xycoords='axes fraction', weight='bold', fontsize=labelticks_size)

    plt.subplots_adjust(top=0.95, bottom=0.085, left=0.115, right=0.975, hspace=0.12, wspace=0.32)
    # plt.show()
    plt.savefig(name_figure, dpi=300)
