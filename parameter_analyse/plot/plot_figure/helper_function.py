#  Copyright 2023 Aix-Marseille Université
# "Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements; and to You under the Apache License, Version 2.0. "
import json
import numpy as np
from scipy import signal
import parameter_analyse.zerlaut_oscilation.python_file.run.tools_simulation as tools
import matplotlib.pyplot as plt


def slidding_window(data, width, variance=False):
    """
    use for mean field
    :param data: instantaneous firing rate
    :param width: windows or times average of the mean field
    :param variance: boolean for getting the variance
    :return: state variable of the mean field
    """
    res = np.zeros((data.shape[0] - width, width))
    res[:, :] = np.squeeze(data[np.array([[i + j for i in range(width)] for j in range(data.shape[0] - width)])])
    if variance:
        return res.mean(axis=1), res.var(axis=1)
    else:
        return res.mean(axis=1)


def get_spike_trains(path, begin, end):
    """
    get spike trains for the reset of the analysis
    :param path: path of spike trains
    :param begin: begin plotting
    :param end: end plotting
    :return: spike trains with 2 extra spikes for begin and end
    """
    spikes_concat = np.loadtxt(path)
    if spikes_concat.shape[0] == 0:
        spikes_concat = np.array([])
        pass
    elif len(spikes_concat.shape) == 1:
        spikes_concat = spikes_concat[1]
    else:
        spikes_concat = spikes_concat[:, 1]
        spikes_concat = spikes_concat[
            np.where(np.logical_and(spikes_concat >= begin, spikes_concat <= end))]
    spikes_concat = np.concatenate(([begin], spikes_concat, [end]))
    return spikes_concat


def get_histogram_frequency(spike_trains, begin, end, nb_neurons, resolution, window_size,
                            frequency=False, variance=False):
    """
    get histogram of a spike_trains
    :param spike_trains: spike trains
    :param begin: start time of data
    :param end: end time of data
    :param nb_neurons: number of neurons
    :param resolution: resolution of the histogram
    :param window_size: window size of the smoothing histogram
    :param frequency: if require return of frequency
    :param variance: boolean for getting the variance
    :return: histogram, frequency
    """
    histogram = np.array(np.histogram(spike_trains, bins=int((end - begin) / resolution)), dtype=object)
    histogram[0] = histogram[0] / (nb_neurons * resolution * 1e-3)
    histogram[0][0] -= 1
    histogram[0][-1] -= 1
    histogram[0][np.where(histogram[0] < 0)] = 0.0
    if window_size == resolution:
        time_network = histogram[1][1:]
        data = histogram[0]
        if variance:
            variance_data = np.zeros_like(histogram[0])
        else:
            variance_data = None
    else:
        time_network = histogram[1][(int(window_size / resolution) + 1):]
        data = slidding_window(histogram[0], int(window_size / resolution), variance=variance)
        if variance:
            variance_data = data[1]
            data = data[0]
        else:
            variance_data = None
    if frequency:
        # frequency
        f, Pxx_den = signal.welch(data, fs=1 / resolution * 1e3,
                                  nperseg=int((end - begin) / resolution) - int(window_size / resolution))
        return time_network, data, variance_data, f, Pxx_den
    else:
        return time_network, data, variance_data


def get_min_max(max_E, min_E, data_ex, M_rateE, variance_ex=None, M_rateE_var=None):
    """
    define the min and the max in function of the variance and rate
    :param max_E: maximum if already provided in other case None
    :param min_E: minimum if already provided in other case None
    :param data_ex: mean firing rate of spiking data
    :param M_rateE: value of the mean field
    :param variance_ex: variance of the mean field
    :param M_rateE_var: variance of the mean field
    :return:
    """
    max_E_define = max_E is None
    min_E_define = min_E is None
    if variance_ex is not None and M_rateE_var is not None:
        max_E = (data_ex + variance_ex).max() if max_E is None else max_E
        min_E = (data_ex - variance_ex).min() if min_E is None else min_E
        for index in range(len(M_rateE)):
            max_E_tmp = (M_rateE[index] + M_rateE_var[index]).max()
            min_E_tmp = (M_rateE[index] - M_rateE_var[index]).min()
            if max_E_tmp > max_E and max_E_define:
                max_E = max_E_tmp
            if min_E > min_E_tmp and min_E_define:
                min_E = min_E_tmp
    else:
        max_E = (data_ex).max() if max_E is None else max_E
        min_E = (data_ex).min() if min_E is None else min_E
        for index in range(len(M_rateE)):
            max_E_tmp = (M_rateE[index]).max()
            min_E_tmp = (M_rateE[index]).min()
            if max_E_tmp > max_E and max_E_define:
                max_E = max_E_tmp
            if min_E > min_E_tmp and min_E_define:
                min_E = min_E_tmp
    return max_E, min_E


def plot_noise_frequency(amplitude, frequency, rate, begin, end, nb_neurons=10 ** 4, ratio_inhibitory=0.2, dt=0.1,
                         zoom_frequency=250, linewidth=0.5,
                         path_result_mean_field=None,
                         path_result_network=None,
                         spike_trains=[(0.2, 0.2), (5.0, 0.2)],
                         mean_field=[(0.0, '/deterministe/'), (1e-9, '/stochastic_1e-09/'),
                                     (1e-8, '/stochastic_1e-08/')],
                         figsize=(20, 20), labelsize=25, ticks_size=20,
                         top=0.92, bottom=0.04, left=0.06, right=0.99, wspace=0.280, hspace=0.5
                         ):
    """
    plot comparison for one specific configuration of the input
    compare the raw data, the smooth data of spiking network with the mean field with and without noise.
    The spectogram is on two graph because the second is a zoom on the first value
    :param amplitude: amplitude of the sinusoidal of the input signal
    :param frequency: frequency of the sinusoidal
    :param rate: mean value of the input signal
    :param begin: start of the plotting simulation
    :param end: end of the plotting simulation
    :param nb_neurons: number of neurons
    :param ratio_inhibitory: ratio of inhibitory neurons
    :param dt: precision of the mean field
    :param zoom_frequency: zoom of the spectogram of the frequency
    :param linewidth: width of the lines
    :param path_result_mean_field: path of the simulation of the mean field
    :param path_result_network: path of the simulation of the network result
    :param spike_trains: define the resolution of the histograms and the smoothing curve
                        [(resolution of the signal, resolution of the histogram), ...]
    :param mean_field: define the noise and the associate folder
                        [(noise variance, '/folder/'), ...]
    :param figsize: figure size
    :param labelsize: label size
    :param ticks_size: ticks label size
    :param top: top marge
    :param bottom: bottom marge
    :param left: left marge
    :param right: right marge
    :param wspace: vertical space
    :param hspace: horizontal space
    :return:
    """
    print('rate_' + str(rate) + '_frequency_' + str(frequency) + '_amplitude_' + str(amplitude))
    ## prapration of the figure
    plt.figure(figsize=figsize)
    plt.suptitle(' Input rate : ' + str(rate) + ' frequency : ' + str(frequency) + '\namplitude :' + str(amplitude)
                 + ' begin :' + str(begin) + ' end:' + str(end), fontsize=labelsize)
    nb_fig = len(spike_trains) + len(mean_field)

    ## spiking network simulation
    # number of neurons by population
    nb_excitatory = int(nb_neurons * (1 - ratio_inhibitory))
    nb_inbitory = int(nb_neurons * ratio_inhibitory)

    path_simulation_network = path_result_network + '/rate_' + str(rate) + '/_frequency_' + str(frequency) \
                              + '_amplitude_' + str(np.around(amplitude, 2)) + '/'
    # get spike data
    spikes_concat_ex = get_spike_trains(path_simulation_network + "/spike_recorder_ex.dat", begin, end)
    spikes_concat_in = get_spike_trains(path_simulation_network + "/spike_recorder_in.dat", begin, end)

    # get histogram and plot it
    for index, (window_size, resolution) in enumerate(spike_trains):
        ax_data = plt.subplot(nb_fig, 3, index * 3 + 1)
        ax_freq = plt.subplot(nb_fig, 3, index * 3 + 2)
        ax_freq_zoom = plt.subplot(nb_fig, 3, index * 3 + 3)
        for spike_train, nb_neurons in [(spikes_concat_ex, nb_excitatory), (spikes_concat_in, nb_inbitory)]:
            time_network, data, variance, f, Pxx_den = get_histogram_frequency(spike_train, begin, end, nb_neurons,
                                                                               resolution, window_size, frequency=True)
            # plot
            ax_data.plot(data, linewidth=linewidth)
            ax_freq.semilogy(f, Pxx_den)
            ax_freq_zoom.semilogy(f[:zoom_frequency], Pxx_den[:zoom_frequency])
        # setup the graph
        ax_data.set_xlabel('ms', fontdict={'fontsize': labelsize})
        title = 'nb spike/' + str(resolution) + 'ms' if resolution == window_size else 'mean spike/' \
                                                                                       + str(window_size) + 'ms'
        ax_data.set_ylabel(title, fontdict={'fontsize': labelsize})
        if window_size == resolution:
            title = 'histogram with resolution of ' + str(resolution) + ' ms'
        else:
            title = 'average mean of the histogram with resolution of ' + str(resolution) + ' ms and a window of 5ms'
        ax_data.tick_params(labelsize=ticks_size)
        ax_freq.set_title(title, fontdict={'fontsize': labelsize})
        ax_freq.set_xlabel('frequency', fontdict={'fontsize': labelsize})
        ax_freq.set_ylabel('power spectrum', fontdict={'fontsize': labelsize})
        ax_freq.tick_params(labelsize=ticks_size)
        ax_freq_zoom.set_xlabel('frequency', fontdict={'fontsize': labelsize})
        ax_freq_zoom.set_ylabel('power spectrum', fontdict={'fontsize': labelsize})
        ax_freq_zoom.tick_params(labelsize=ticks_size)

    index_graph = len(spike_trains) * 3 + 1
    for index, (variance_noise, path_mean_field) in enumerate(mean_field):
        # get data
        path_simulation_mean_field = path_result_mean_field + path_mean_field + '/rate_' + str(rate) \
                                     + '/frequency_' + str(frequency)
        with open(path_simulation_mean_field + '/parameter.json') as f:
            parameters = json.load(f)
        amplitudes = np.array(parameters['parameter_stimulation']["amp"]) * 1e3
        amplitude_index = np.where(amplitudes == amplitude)[0][0]

        result_mean_field_1 = tools.get_result(path_simulation_mean_field, begin, end)
        M_rateE = result_mean_field_1[0][1][:, 0, amplitude_index] * 1e3
        M_rateI = result_mean_field_1[0][1][:, 1, amplitude_index] * 1e3
        f_ex, Pxx_den_ex = signal.welch(M_rateE, fs=1 / dt * 1e3, nperseg=int((end - begin - 1) / dt) - 1)
        f_in, Pxx_den_in = signal.welch(M_rateI, fs=1 / dt * 1e3, nperseg=int((end - begin - 1) / dt) - 1)

        # plot data
        ax = plt.subplot(nb_fig, 3, index_graph + index * 3)
        ax.plot(M_rateE, linewidth=linewidth)
        ax.plot(M_rateI, linewidth=linewidth)
        ax.set_xlabel('ms', fontdict={'fontsize': labelsize})
        ax.set_ylabel('mean firing rate', fontdict={'fontsize': labelsize})
        ax.tick_params(labelsize=ticks_size)

        ax = plt.subplot(nb_fig, 3, index_graph + 1 + index * 3)
        ax.semilogy(f_ex, Pxx_den_ex)
        ax.semilogy(f_in, Pxx_den_in)
        ax.set_title('Mean field with noise variance ' + str(variance_noise), fontdict={'fontsize': labelsize})
        ax.set_xlabel('frequency', fontdict={'fontsize': labelsize})
        ax.set_ylabel('power spectrum   ', fontdict={'fontsize': labelsize})
        ax.tick_params(labelsize=ticks_size)

        ax = plt.subplot(nb_fig, 3, index_graph + 2 + index * 3)
        ax.semilogy(f_ex[:zoom_frequency], Pxx_den_ex[:zoom_frequency])
        ax.semilogy(f_in[:zoom_frequency], Pxx_den_in[:zoom_frequency])
        ax.set_xlabel('frequency', fontdict={'fontsize': labelsize})
        ax.set_ylabel('power spectrum   ', fontdict={'fontsize': labelsize})
        ax.tick_params(labelsize=ticks_size)

    plt.subplots_adjust(top=top, bottom=bottom, left=left, right=right, wspace=wspace, hspace=hspace)


def plot_compare_frequence(amplitude, frequency, rate, begin, end, nb_neurons=10 ** 4, ratio_inhibitory=0.2, dt=0.1,
                           zoom_frequency=250, linewidth=[0.5, 0.5],
                           path_result_mean_field=None,
                           path_result_network=None,
                           resolution=0.2, window_size=0.2, path_mean_field='/deterministe/', plot_excitatory=True,
                           plot_inhibitory=False, variance_plot=False,
                           figsize=(20, 20), labelsize=30, ticks_size=20,
                           top=0.920, bottom=0.050, left=0.070, right=0.990, wspace=0.180, hspace=0.6
                           ):
    """
    plot a comparison between 2 results with frequency
    :param amplitude: amplitude of the sinusoidal of the input signal
    :param frequency: frequency of the sinusoidal
    :param rate: mean value of the input signal
    :param begin: start of the plotting simulation
    :param end: end of the plotting simulation
    :param nb_neurons: number of neurons
    :param ratio_inhibitory: ratio of inhibitory neurons
    :param dt: precision of the mean field
    :param zoom_frequency: zoom of the spectogram of the frequency
    :param linewidth: array for width of the lines of the mean field and the network
    :param path_result_mean_field: path of the simulation of the mean field
    :param path_result_network: path of the simulation of the network result
    :param resolution: resolution of the histogram
    :param window_size: window for smoothing the histogram
    :param path_mean_field: folder with mean field result
    :param plot_excitatory: plot excitatory neurons
    :param plot_inhibitory: plot for inhibitory neurons
    :param variance_plot: plot the variance or not
    :param figsize: figure size
    :param labelsize: label size
    :param ticks_size: ticks label size
    :param top: top marge
    :param bottom: bottom marge
    :param left: left marge
    :param right: right marge
    :param wspace: vertical space
    :param hspace: horizontal space
    :return:
    """
    print('rate_' + str(rate) + '_frequency_' + str(frequency) + '_amplitude_' + str(amplitude))
    ### get data
    ## spiking network simulation
    # pat of the simulation
    path_simulation_network = path_result_network + '/rate_' + str(rate) + '/_frequency_' + str(frequency) \
                              + '_amplitude_' + str(np.around(amplitude, 2)) + '/'
    # path of the mean field
    path_simulation_mean_field = path_result_mean_field + path_mean_field + \
                                 '/rate_' + str(rate) + '/frequency_' + str(frequency)
    # get data of the mean field
    with open(path_simulation_mean_field + '/parameter.json') as f:
        parameters = json.load(f)
    amplitudes = np.array(parameters['parameter_stimulation']["amp"]) * 1e3
    amplitude_index = np.where(amplitudes == amplitude)[0][0]
    result_mean_field = tools.get_result(path_simulation_mean_field, begin, end)
    time_mean_field = np.array(result_mean_field[0][0], dtype=float)
    M_rateE = result_mean_field[0][1][:, 0, amplitude_index] * 1e3
    M_rateE_var = result_mean_field[0][1][:, 2, amplitude_index] * 1e6
    M_rateI = result_mean_field[0][1][:, 1, amplitude_index] * 1e3
    M_rateI_var = result_mean_field[0][1][:, 4, amplitude_index] * 1e6

    if plot_excitatory:
        # get excitatory spike of th network
        nb_excitatory = int(nb_neurons * (1 - ratio_inhibitory))
        spikes_concat_ex = get_spike_trains(path_simulation_network + "/spike_recorder_ex.dat", begin, end)
        time_network, data_ex, variance_ex, f_ex, Pxx_den_ex = get_histogram_frequency(
            spikes_concat_ex, begin, end, nb_excitatory, resolution, window_size, frequency=True,
            variance=variance_plot)
        M_f_ex, M_Pxx_den_ex = signal.welch(M_rateE, fs=1 / dt * 1e3, nperseg=int((end - begin - 1) / dt) - 1)

        # plot figure
        plt.figure(figsize=figsize)
        plt.suptitle('Excitatory: Input rate : ' + str(rate) + ' frequency : ' + str(frequency) +
                     '\n amplitude :' + str(amplitude) + ' begin :' + str(begin) + ' end:' + str(end),
                     fontsize=labelsize)
        # plot curve
        ax = plt.subplot(1, 2, 1)
        ax.plot(time_network, data_ex, linewidth=linewidth[0], color='b')
        ax.plot(time_mean_field, M_rateE, linewidth=linewidth[1], color='orange')
        if variance_plot:
            ax.fill_between(time_network, data_ex + variance_ex, data_ex - variance_ex, linewidth=linewidth[0],
                            color='b', alpha=0.2)
            ax.fill_between(time_mean_field, M_rateE + M_rateE_var, M_rateE - M_rateE_var, linewidth=linewidth[1],
                            color='orange', alpha=0.2)
        ax.set_xlabel('ms', fontdict={'fontsize': labelsize})
        title = 'nb spike/' + str(resolution) + 'ms' if resolution == window_size else 'mean spike/' \
                                                                                       + str(window_size) + 'ms'
        ax.set_ylabel(title, fontdict={'fontsize': labelsize})
        ax.tick_params(labelsize=ticks_size)
        # plot frequency
        ax = plt.subplot(1, 2, 2)
        ax.semilogy(f_ex[:zoom_frequency], Pxx_den_ex[:zoom_frequency], label='network', color='b')
        ax.semilogy(M_f_ex[:zoom_frequency], M_Pxx_den_ex[:zoom_frequency], label='mean field', color='orange')
        ax.set_xlabel('frequency', fontdict={'fontsize': labelsize})
        ax.set_ylabel('power spectrum', fontdict={'fontsize': labelsize})
        ax.tick_params(labelsize=ticks_size)
        plt.legend(fontsize=ticks_size)
        plt.subplots_adjust(top=top, bottom=bottom, left=left, right=right, wspace=wspace, hspace=hspace)

    if plot_inhibitory:
        # get inhibitory spike of th network
        nb_inhibitory = int(nb_neurons * ratio_inhibitory)
        spikes_concat_in = get_spike_trains(path_simulation_network + "/spike_recorder_in.dat", begin, end)
        time_network, data_in, variance_in, f_in, Pxx_den_in = get_histogram_frequency(
            spikes_concat_in, begin, end, nb_inhibitory, resolution, window_size, frequency=True,
            variance=variance_plot)
        M_f_in, M_Pxx_den_in = signal.welch(M_rateI, fs=1 / dt * 1e3, nperseg=int((end - begin - 1) / dt) - 1)

        # plot figure
        plt.figure(figsize=figsize)
        plt.suptitle('Inhibitory: Input rate : ' + str(rate) + ' frequency : ' + str(frequency) +
                     '\namplitude :' + str(amplitude) + ' begin :' + str(begin) + ' end:' + str(end),
                     fontsize=labelsize)
        # plot curve
        ax = plt.subplot(1, 2, 1)
        ax.plot(time_network, data_in, linewidth=linewidth[0], color='b')
        ax.plot(time_mean_field, M_rateI, linewidth=linewidth[1], color='orange')
        if variance_plot:
            ax.fill_between(time_network, data_in + variance_in, data_in - variance_in, linewidth=linewidth[0],
                            color='b', alpha=0.2)
            ax.fill_between(time_mean_field, M_rateI + M_rateI_var, M_rateI - M_rateI_var, linewidth=linewidth[1],
                            color='orange', alpha=0.2)
        ax.set_xlabel('ms', fontdict={'fontsize': labelsize})
        title = 'nb spike/' + str(resolution) + 'ms' if resolution == window_size else 'mean spike/' + \
                                                                                       str(window_size) + 'ms'
        ax.set_ylabel(title, fontdict={'fontsize': labelsize})
        ax.tick_params(labelsize=ticks_size)
        # plot frequency
        ax = plt.subplot(1, 2, 2)
        ax.semilogy(f_in[:zoom_frequency], Pxx_den_in[:zoom_frequency], label='network')
        ax.semilogy(M_f_in[:zoom_frequency], M_Pxx_den_in[:zoom_frequency], label='mean field')
        ax.set_xlabel('frequency', fontdict={'fontsize': labelsize})
        ax.set_ylabel('power spectrum', fontdict={'fontsize': labelsize})
        ax.tick_params(labelsize=ticks_size)
        plt.legend(fontsize=ticks_size)
        plt.subplots_adjust(top=top, bottom=bottom, left=left, right=right, wspace=wspace, hspace=hspace)


def plot_compare_frequence_noise(amplitude, frequency, rate, begin, end, nb_neurons=10 ** 4, ratio_inhibitory=0.2,
                                 dt=0.1, zoom_frequency=250, linewidth=[0.5, 0.5],
                                 path_result_mean_field=None,
                                 path_result_network=None,
                                 resolution=0.2, window_size=0.2, path_mean_fields=['/deterministe/', ],
                                 plot_excitatory=True, plot_inhibitory=False,
                                 color_curve_meanfield=['orange', 'green', 'red'],
                                 variance_plot=False, max_E=None, min_E=None, max_I=None, min_I=None,
                                 figsize=(20, 20), labelsize=30, ticks_size=20,
                                 top=0.92, bottom=0.05, left=0.06, right=0.99, wspace=0.20, hspace=0.1
                                 ):
    """
    plot frequency noise on the the same graphic
    :param amplitude: amplitude of the sinusoidal of the input signal
    :param frequency: frequency of the sinusoidal
    :param rate: mean value of the input signal
    :param begin: start of the plotting simulation
    :param end: end of the plotting simulation
    :param nb_neurons: number of neurons
    :param ratio_inhibitory: ratio of inhibitory neurons
    :param dt: precision of the mean field
    :param zoom_frequency: zoom of the spectogram of the frequency
    :param linewidth: array for width of the lines of the mean field and the network
    :param path_result_mean_field: path of the simulation of the mean field
    :param path_result_network: path of the simulation of the network result
    :param resolution: resolution of the histogram
    :param window_size: window for smoothing the histogram
    :param path_mean_fields: folder with mean field result
    :param plot_excitatory: plot excitatory neurons
    :param plot_inhibitory: plot for inhibitory neurons
    :param color_curve_meanfield: color for curve of the mean field
    :param variance_plot: plot the variance or not
    :param max_E: max cut the graphic for mean firing rate of the excitatory
    :param min_E: min cut the graphic for mean firing rate of the excitatory
    :param max_I: max cut the graphic for mean firing rate of the inhibitory
    :param min_I: min cut the graphic for mean firing rate of the inhibitory
    :param figsize: figure size
    :param labelsize: label size
    :param ticks_size: ticks label size
    :param top: top marge
    :param bottom: bottom marge
    :param left: left marge
    :param right: right marge
    :param wspace: vertical space
    :param hspace: horizontal space
    :return:
    """
    print('rate_' + str(rate) + '_frequency_' + str(frequency) + '_amplitude_' + str(amplitude))

    # path for spike trains
    path_simulation_network = path_result_network + '/rate_' + str(rate) + '/_frequency_' + str(
        frequency) + '_amplitude_' + str(np.around(amplitude, 2)) + '/'

    # get data from mean field
    M_rateE = []
    M_rateE_var = []
    M_rateI = []
    M_rateI_var = []
    time_mean_field = None
    for path_mean_field in path_mean_fields:
        path_simulation_mean_field = path_result_mean_field + '/' + path_mean_field + '/' + \
                                     '/rate_' + str(rate) + '/frequency_' + str(frequency)
        with open(path_simulation_mean_field + '/parameter.json') as f:
            parameters = json.load(f)
        amplitudes = np.array(parameters['parameter_stimulation']["amp"]) * 1e3
        amplitude_index = np.where(amplitudes == amplitude)[0][0]
        result_mean_field = tools.get_result(path_simulation_mean_field, begin, end)
        time_mean_field = np.array(result_mean_field[0][0], dtype=float)
        M_rateE.append(result_mean_field[0][1][:, 0, amplitude_index] * 1e3)
        M_rateE_var.append(result_mean_field[0][1][:, 2, amplitude_index] * 1e6)
        M_rateI.append(result_mean_field[0][1][:, 1, amplitude_index] * 1e3)
        M_rateI_var.append(result_mean_field[0][1][:, 4, amplitude_index] * 1e6)

    if plot_excitatory:
        # get excitatory spike of th network
        nb_excitatory = int(nb_neurons * (1 - ratio_inhibitory))
        spikes_concat_ex = get_spike_trains(path_simulation_network + "/spike_recorder_ex.dat", begin, end)
        time_network, data_ex, variance_ex, f_ex, Pxx_den_ex = get_histogram_frequency(
            spikes_concat_ex, begin, end, nb_excitatory, resolution, window_size, frequency=True,
            variance=variance_plot)
        M_f_ex = []
        for index in range(len(M_rateE)):
            M_f_ex.append(signal.welch(M_rateE[index], fs=1 / dt * 1e3, nperseg=int((end - begin - 1) / dt) - 1))
        max_E, min_E = get_min_max(max_E, min_E, data_ex, M_rateE, variance_ex, M_rateE_var)

        # plot figure
        plt.figure(figsize=figsize)
        plt.suptitle('Excitatory: Input rate : ' + str(rate) + ' frequency : ' + str(frequency)
                     + '\namplitude :' + str(amplitude) + ' begin :' + str(begin) + ' end:' + str(end),
                     fontsize=labelsize)
        # plot network data
        ax = plt.subplot(2, 1, 1)
        ax.plot(time_network, data_ex, linewidth=linewidth[0])
        if variance_plot:
            ax.fill_between(time_network, data_ex + variance_ex, data_ex[0] - variance_ex, linewidth=linewidth[0],
                            color='b', alpha=0.2)
        # plot of the different mean field
        for index in range(len(M_rateE)):
            ax.plot(time_mean_field, M_rateE[index], '--', linewidth=linewidth[1], color=color_curve_meanfield[index],
                    label=path_mean_fields[index])
            if variance_plot:
                ax.fill_between(time_mean_field, M_rateE[index] + M_rateE_var[index],
                                M_rateE[index] - M_rateE_var[index], linewidth=linewidth[1],
                                color=color_curve_meanfield[index], alpha=0.2)
        ax.set_xlabel('ms', fontdict={'fontsize': labelsize})
        title = 'nb spike/' + str(resolution) + 'ms' if resolution == window_size else 'mean spike/' \
                                                                                       + str(window_size) + 'ms'
        ax.set_ylabel(title, fontdict={'fontsize': labelsize})
        ax.tick_params(labelsize=ticks_size)
        ax.set_ylim(ymax=max_E, ymin=min_E)

        # plot spectogramme
        ax = plt.subplot(2, 1, 2)
        ax.semilogy(f_ex[:zoom_frequency], Pxx_den_ex[:zoom_frequency], label='network', linewidth=linewidth[0])
        for index, (f, Pxx_den) in enumerate(M_f_ex):
            ax.semilogy(f[:zoom_frequency], Pxx_den[:zoom_frequency], '--',
                        label=path_mean_fields[index], linewidth=linewidth[1])
        ax.set_xlabel('frequency', fontdict={'fontsize': labelsize})
        ax.set_ylabel('power spectrum', fontdict={'fontsize': labelsize})
        ax.tick_params(labelsize=ticks_size)
        plt.legend(fontsize=ticks_size)
        plt.subplots_adjust(top=top, bottom=bottom, left=left, right=right, wspace=wspace, hspace=hspace)

    if plot_inhibitory:
        # get inhibitory spike of th network
        nb_inhibitory = int(nb_neurons * ratio_inhibitory)
        spikes_concat_in = get_spike_trains(path_simulation_network + "/spike_recorder_in.dat", begin, end)
        time_network, data_in, variance_in, f_in, Pxx_den_in = get_histogram_frequency(
            spikes_concat_in, begin, end, nb_inhibitory, resolution, window_size, frequency=True,
            variance=variance_plot)

        M_f_in = []
        for index in range(len(M_rateI)):
            M_f_in.append(signal.welch(M_rateI[index], fs=1 / dt * 1e3, nperseg=int((end - begin - 1) / dt) - 1))
        max_I, min_I = get_min_max(max_I, min_I, data_in, M_rateI, variance_in, M_rateI_var)

        # plot figure
        plt.figure(figsize=figsize)
        plt.suptitle('Inhibitory: Input rate : ' + str(rate) + ' frequency : ' + str(frequency)
                     + '\namplitude :' + str(amplitude) + ' begin :' + str(begin) + ' end:' + str(end),
                     fontsize=labelsize)
        # plot network data
        ax = plt.subplot(2, 1, 1)
        ax.plot(time_network, data_in, linewidth=linewidth[0])
        if variance_plot:
            ax.fill_between(time_network, data_in + variance_in, data_in[0] - variance_in, linewidth=linewidth[0],
                            color='b', alpha=0.2)
        # plot of the different mean field
        for index in range(len(M_rateI)):
            ax.plot(time_mean_field, M_rateI[index], '--', linewidth=linewidth[1], color=color_curve_meanfield[index],
                    label=path_mean_fields[index])
            if variance_plot:
                ax.fill_between(time_mean_field, M_rateI[index] + M_rateI_var[index],
                                M_rateI[index] - M_rateI_var[index], linewidth=linewidth[1],
                                color=color_curve_meanfield[index], alpha=0.2)
        ax.set_xlabel('ms', fontdict={'fontsize': labelsize})
        title = 'nb spike/' + str(resolution) + 'ms' if resolution == window_size else 'mean spike/' \
                                                                                       + str(window_size) + 'ms'
        ax.set_ylabel(title, fontdict={'fontsize': labelsize})
        ax.tick_params(labelsize=ticks_size)
        ax.set_ylim(ymax=max_I, ymin=min_I)

        # plot spectogramme
        ax = plt.subplot(2, 1, 2)
        ax.semilogy(f_in[:zoom_frequency], Pxx_den_in[:zoom_frequency], label='network', linewidth=linewidth[0])
        for index, (f, Pxx_den) in enumerate(M_f_in):
            ax.semilogy(f[:zoom_frequency], Pxx_den[:zoom_frequency], '--',
                        label=path_mean_fields[index], linewidth=linewidth[1])
        ax.set_xlabel('frequency', fontdict={'fontsize': labelsize})
        ax.set_ylabel('power spectrum', fontdict={'fontsize': labelsize})
        ax.tick_params(labelsize=ticks_size)
        plt.legend(fontsize=ticks_size)
        plt.subplots_adjust(top=top, bottom=bottom, left=left, right=right, wspace=wspace, hspace=hspace)


def plot_compare(amplitude, frequency, rate, begin, end, nb_neurons=10 ** 4, ratio_inhibitory=0.2, linewidth=[0.5, 1.0],
                 path_result_mean_field=None,
                 path_result_network=None,
                 resolution=0.2, window_size=0.2, path_mean_field='/deterministe/', plot_excitatory=True,
                 plot_inhibitory=False, variance_plot=False,
                 figsize=(20, 20), labelsize=30, ticks_size=20,
                 top=0.92, bottom=0.04, left=0.05, right=0.99, wspace=0.20, hspace=0.23
                 ):
    """
    plot comparison of the 1 mean field
    :param amplitude: amplitude of the sinusoidal of the input signal
    :param frequency: frequency of the sinusoidal
    :param rate: mean value of the input signal
    :param begin: start of the plotting simulation
    :param end: end of the plotting simulation
    :param nb_neurons: number of neurons
    :param ratio_inhibitory: ratio of inhibitory neurons
    :param dt: precision of the mean field
    :param zoom_frequency: zoom of the spectogram of the frequency
    :param linewidth: array for width of the lines of the mean field and the network
    :param path_result_mean_field: path of the simulation of the mean field
    :param path_result_network: path of the simulation of the network result
    :param resolution: resolution of the histogram
    :param window_size: window for smoothing the histogram
    :param path_mean_field: folder with mean field result
    :param plot_excitatory: plot excitatory neurons
    :param plot_inhibitory: plot for inhibitory neurons
    :param variance_plot: plot the variance or nod and the network
    :param figsize: figure size
    :param labelsize: label size
    :param ticks_size: ticks label size
    :param top: top marge
    :param bottom: bottom marge
    :param left: left marge
    :param right: right marge
    :param wspace: vertical space
    :param hspace: horizontal space
    :return:
    """
    print('rate_' + str(rate) + '_frequency_' + str(frequency) + '_amplitude_' + str(amplitude))
    path_simulation_network = path_result_network + '/rate_' + str(rate) + '/_frequency_' + str(
        frequency) + '_amplitude_' + str(np.around(amplitude, 2)) + '/'

    path_simulation_mean_field = path_result_mean_field + path_mean_field + \
                                 '/rate_' + str(rate) + '/frequency_' + str(frequency)
    with open(path_simulation_mean_field + '/parameter.json') as f:
        parameters = json.load(f)
    amplitudes = np.array(parameters['parameter_stimulation']["amp"]) * 1e3
    amplitude_index = np.where(amplitudes == amplitude)[0][0]
    result_mean_field = tools.get_result(path_simulation_mean_field, begin, end)
    time_mean_field = np.array(result_mean_field[0][0], dtype=float)
    M_rateE = result_mean_field[0][1][:, 0, amplitude_index] * 1e3
    M_rateE_var = result_mean_field[0][1][:, 2, amplitude_index] * 1e6
    M_rateI = result_mean_field[0][1][:, 1, amplitude_index] * 1e3
    M_rateI_var = result_mean_field[0][1][:, 4, amplitude_index] * 1e6

    if plot_excitatory:
        # get excitatory spike of the network
        nb_excitatory = int(nb_neurons * (1 - ratio_inhibitory))
        spikes_concat_ex = get_spike_trains(path_simulation_network + "/spike_recorder_ex.dat", begin, end)
        time_network, data_ex, variance_ex, f_ex, Pxx_den_ex = get_histogram_frequency(
            spikes_concat_ex, begin, end, nb_excitatory, resolution, window_size, frequency=True,
            variance=variance_plot)
        # plot figure
        plt.figure(figsize=figsize)
        plt.suptitle('Excitatory: Input rate : ' + str(rate) + ' frequency : ' + str(frequency)
                     + '\namplitude :' + str(amplitude) + ' begin :' + str(begin) + ' end:' + str(end),
                     fontsize=labelsize)
        ax = plt.subplot(1, 1, 1)
        ax.plot(time_network, data_ex, linewidth=linewidth[0], color='b')
        ax.plot(time_mean_field, M_rateE, linewidth=linewidth[1], color='orange')
        if variance_plot:
            ax.fill_between(time_network, data_ex + variance_ex, data_ex - variance_ex, linewidth=linewidth[0],
                            color='b', alpha=0.2)
            ax.fill_between(time_mean_field, M_rateE + M_rateE_var, M_rateE - M_rateE_var, linewidth=linewidth[1],
                            color='orange', alpha=0.2)
        ax.set_xlabel('ms', fontdict={'fontsize': labelsize})
        title = 'nb spike/' + str(resolution) + 'ms' if resolution == window_size else 'mean spike/' \
                                                                                       + str(window_size) + 'ms'
        ax.set_ylabel(title, fontdict={'fontsize': labelsize})
        ax.tick_params(labelsize=ticks_size)
        plt.legend(fontsize=ticks_size)
        plt.subplots_adjust(top=top, bottom=bottom, left=left, right=right, wspace=wspace, hspace=hspace)

    if plot_inhibitory:
        # get inhibitory spike of the network
        nb_inhibitory = int(nb_neurons * ratio_inhibitory)
        spikes_concat_in = get_spike_trains(path_simulation_network + "/spike_recorder_in.dat", begin, end)
        time_network, data_in, variance_in, f_in, Pxx_den_in = get_histogram_frequency(
            spikes_concat_in, begin, end, nb_inhibitory, resolution, window_size, frequency=True,
            variance=variance_plot)

        # plot figure
        plt.figure(figsize=figsize)
        plt.suptitle('Inhibitory: Input rate : ' + str(rate) + ' frequency : ' + str(frequency)
                     + '\namplitude :' + str(amplitude) + ' begin :' + str(begin) + ' end:' + str(end),
                     fontsize=labelsize)
        ax = plt.subplot(1, 1, 1)
        ax.plot(time_network, data_in, linewidth=linewidth[0], color='b')
        ax.plot(time_mean_field, M_rateI, linewidth=linewidth[1], color='orange')
        if variance_plot:
            ax.fill_between(time_network, data_in + variance_in, data_in - variance_in, linewidth=linewidth[0],
                            color='b', alpha=0.2)
            ax.fill_between(time_mean_field, M_rateI + M_rateI_var, M_rateI - M_rateI_var, linewidth=linewidth[1],
                            color='orange', alpha=0.2)
        ax.set_xlabel('ms', fontdict={'fontsize': labelsize})
        title = 'nb spike/' + str(resolution) + 'ms' if resolution == window_size else 'mean spike/' \
                                                                                       + str(window_size) + 'ms'
        ax.set_ylabel(title, fontdict={'fontsize': labelsize})
        plt.legend(fontsize=ticks_size)
        plt.subplots_adjust(top=top, bottom=bottom, left=left, right=right, wspace=wspace, hspace=hspace)


def plot_compare_all(amplitude, frequency, rate, begin, end, nb_neurons=10 ** 4, ratio_inhibitory=0.2, dt=0.1,
                     linewidth=[0.5, 1.0],
                     path_result_mean_field=None,
                     path_result_network=None,
                     resolution=0.2, window_size=0.2, path_mean_fields=['/deterministe/'], plot_excitatory=True,
                     plot_inhibitory=False, variance_plot=False, max_E=None, min_E=None, max_I=None, min_I=None,
                     color_curve_meanfield=['orange', 'green', 'red'],
                     figsize=(20, 20), labelsize=30, ticks_size=20,
                     top=0.92, bottom=0.04, left=0.04, right=0.99, wspace=0.20, hspace=0.35
                     ):
    """
    plt compare different noise and frequency on the same graphs
    :param amplitude: amplitude of the sinusoidal of the input signal
    :param frequency: frequency of the sinusoidal
    :param rate: mean value of the input signal
    :param begin: start of the plotting simulation
    :param end: end of the plotting simulation
    :param nb_neurons: number of neurons
    :param ratio_inhibitory: ratio of inhibitory neurons
    :param dt: precision of the mean field
    :param zoom_frequency: zoom of the spectogram of the frequency
    :param linewidth: array for width of the lines of the mean field and the network
    :param path_result_mean_field: path of the simulation of the mean field
    :param path_result_network: path of the simulation of the network result
    :param resolution: resolution of the histogram
    :param window_size: window for smoothing the histogram
    :param path_mean_fields: folder with mean field result
    :param plot_excitatory: plot excitatory neurons
    :param plot_inhibitory: plot for inhibitory neurons
    :param color_curve_meanfield: color for curve of the mean field
    :param variance_plot: plot the variance or not
    :param max_E: max cut the graphic for mean firing rate of the excitatory
    :param min_E: min cut the graphic for mean firing rate of the excitatory
    :param max_I: max cut the graphic for mean firing rate of the inhibitory
    :param min_I: min cut the graphic for mean firing rate of the inhibitory
    :param figsize: figure size
    :param labelsize: label size
    :param ticks_size: ticks label size
    :param top: top marge
    :param bottom: bottom marge
    :param left: left marge
    :param right: right marge
    :param wspace: vertical space
    :param hspace: horizontal space
    :return:
    """
    print('rate_' + str(rate) + '_frequency_' + str(frequency) + '_amplitude_' + str(amplitude))
    path_simulation_network = path_result_network + '/rate_' + str(rate) + '/_frequency_' + str(
        frequency) + '_amplitude_' + str(np.around(amplitude, 2)) + '/'

    M_rateE = []
    M_rateE_var = []
    M_rateI = []
    M_rateI_var = []
    time_mean_field = None
    for path_mean_field in path_mean_fields:
        path_simulation_mean_field = path_result_mean_field + '/' + path_mean_field + '/' + \
                                     '/rate_' + str(rate) + '/frequency_' + str(frequency)
        with open(path_simulation_mean_field + '/parameter.json') as f:
            parameters = json.load(f)
        amplitudes = np.array(parameters['parameter_stimulation']["amp"]) * 1e3
        amplitude_index = np.where(amplitudes == amplitude)[0][0]
        result_mean_field = tools.get_result(path_simulation_mean_field, begin, end)
        time_mean_field = np.array(result_mean_field[0][0], dtype=float)
        M_rateE.append(result_mean_field[0][1][:, 0, amplitude_index] * 1e3)
        M_rateE_var.append(result_mean_field[0][1][:, 2, amplitude_index] * 1e6)
        M_rateI.append(result_mean_field[0][1][:, 1, amplitude_index] * 1e3)
        M_rateI_var.append(result_mean_field[0][1][:, 4, amplitude_index] * 1e6)

    if plot_excitatory:
        # get excitatory spike of th network
        nb_excitatory = int(nb_neurons * (1 - ratio_inhibitory))
        spikes_concat_ex = get_spike_trains(path_simulation_network + "/spike_recorder_ex.dat", begin, end)
        time_network, data_ex, variance_ex, f_ex, Pxx_den_ex = get_histogram_frequency(
            spikes_concat_ex, begin, end, nb_excitatory, resolution, window_size, frequency=True,
            variance=variance_plot)
        M_f_ex = []
        for index in range(len(M_rateE)):
            M_f_ex.append(signal.welch(M_rateE[index], fs=1 / dt * 1e3, nperseg=int((end - begin - 1) / dt) - 1))
        max_E, min_E = get_min_max(max_E, min_E, data_ex, M_rateE, variance_ex, M_rateE_var)

        # plot figure
        plt.figure(figsize=figsize)
        plt.suptitle('Excitatory: Input rate : ' + str(rate) + ' frequency : ' + str(frequency)
                     + '\namplitude :' + str(amplitude) + ' begin :' + str(begin) + ' end:' + str(end),
                     fontsize=labelsize)
        ax = plt.subplot(1 + len(M_rateE), 1, 1)
        ax.plot(time_network, data_ex, linewidth=linewidth[0], color='b')
        if variance_plot:
            ax.fill_between(time_network, data_ex + variance_ex, data_ex - variance_ex, linewidth=linewidth[0],
                            color='b', alpha=0.2)
        ax.set_xlabel('ms', fontdict={'fontsize': labelsize})
        title = 'nb spike/' + str(resolution) + 'ms' if resolution == window_size else 'mean spike/' \
                                                                                       + str(window_size) + 'ms'
        ax.set_ylabel(title, fontdict={'fontsize': labelsize})
        ax.tick_params(labelsize=ticks_size)
        ax.set_ylim(ymax=max_E, ymin=min_E)

        for index in range(len(M_rateE)):
            ax = plt.subplot(1 + len(M_rateE), 1, 2 + index)
            ax.plot(time_mean_field, M_rateE[index], linewidth=linewidth[1], color=color_curve_meanfield[index])
            if variance_plot:
                ax.fill_between(time_mean_field, M_rateE[index] + M_rateE_var[index],
                                M_rateE[index] - M_rateE_var[index], linewidth=linewidth[1],
                                color=color_curve_meanfield[index], alpha=0.2)
            ax.set_xlabel('ms', fontdict={'fontsize': labelsize})
            title = 'Hz'
            ax.set_ylabel(title, fontdict={'fontsize': labelsize})
            ax.set_title(path_mean_fields[index], fontdict={'fontsize': labelsize})
            ax.tick_params(labelsize=ticks_size)
            ax.set_ylim(ymax=max_E, ymin=min_E)
        plt.subplots_adjust(top=top, bottom=bottom, left=left, right=right, wspace=wspace, hspace=hspace)

    if plot_inhibitory:
        # get inhibitory spike of the network
        nb_inhibitory = int(nb_neurons * ratio_inhibitory)
        spikes_concat_in = get_spike_trains(path_simulation_network + "/spike_recorder_in.dat", begin, end)
        time_network, data_in, variance_in, f_in, Pxx_den_in = get_histogram_frequency(
            spikes_concat_in, begin, end, nb_inhibitory, resolution, window_size, frequency=True,
            variance=variance_plot)
        M_f_in = []
        for index in range(len(M_rateI)):
            M_f_in.append(signal.welch(M_rateI[index], fs=1 / dt * 1e3, nperseg=int((end - begin - 1) / dt) - 1))
        max_I, min_I = get_min_max(max_I, min_I, data_in, M_rateI, variance_in, M_rateI_var)

        plt.figure(figsize=(20, 20))
        plt.suptitle('Inhibitory: Input rate : ' + str(rate) + ' frequency : ' + str(frequency) +
                     '\namplitude :' + str(amplitude) + ' begin :' + str(begin) + ' end:' + str(end),
                     fontsize=labelsize)
        ax = plt.subplot(1 + len(M_rateI), 1, 1)
        ax.plot(time_network, data_in, linewidth=linewidth[0], color='b')
        if variance_plot:
            ax.fill_between(time_network, data_in + variance_in, data_in - variance_in, linewidth=linewidth[0],
                            color='b', alpha=0.2)
        ax.set_xlabel('ms', fontdict={'fontsize': labelsize})
        title = 'nb spike/' + str(resolution) + 'ms' if resolution == window_size else 'mean spike/' \
                                                                                       + str(window_size) + 'ms'
        ax.set_ylabel(title, fontdict={'fontsize': labelsize})
        ax.tick_params(labelsize=ticks_size)
        ax.set_ylim(ymax=max_I, ymin=min_I)

        for index in range(len(M_rateI)):
            ax = plt.subplot(1 + len(M_rateI), 1, 2 + index)
            ax.plot(time_mean_field, M_rateI[index], linewidth=linewidth[1], color=color_curve_meanfield[index])
            if variance_plot:
                ax.fill_between(time_mean_field, M_rateI[index] + M_rateI_var[index],
                                M_rateI[index] - M_rateI_var[index], linewidth=linewidth[1],
                                color=color_curve_meanfield[index], alpha=0.2)
            else:
                ax.plot(time_mean_field, M_rateI[index], linewidth=linewidth[1])
            ax.set_xlabel('ms', fontdict={'fontsize': labelsize})
            title = 'Hz'
            ax.set_ylabel(title, fontdict={'fontsize': labelsize})
            ax.set_title(path_mean_fields[index], fontdict={'fontsize': labelsize})
            ax.tick_params(labelsize=ticks_size)
            ax.set_ylim(ymax=max_I, ymin=min_I)
        plt.subplots_adjust(top=top, bottom=bottom, left=left, right=right, wspace=wspace, hspace=hspace)
