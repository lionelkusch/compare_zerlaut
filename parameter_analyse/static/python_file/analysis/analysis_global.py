import numpy as np
import os
import json
from elephant.statistics import isi, cv, lv
from elephant.spectral import welch_psd
from scipy.stats import variation

from parameter_analyse.static.python_file.analysis.result_class import Result_analyse


def get_gids(path, number):
    """
    Get the id of the different neurons in the network by population
    :param path: the path to the file
    :param number: the position of the gid population
    :return: an array with the range of id for the different population
    """
    gidfile = open(path + '/population_GIDs.dat', 'r')
    gids = []
    for l in gidfile:
        a = l.split()
        gids.append([int(a[0]), int(a[1]), a[2]])
    return gids[number]


def load_spike(gids, path, begin, end, number):
    """
    Get the id of the neurons which create the spike and time
    :param gids: the array of the id
    :param path: the path to the file
    :param begin: the first time
    :param end: the end of time
    :param number: the position of the gid population
    :return: The spike of all neurons between end and begin
    """
    add = 'ex' if number == 0 else 'in'
    if not os.path.exists(path + "/spike_recorder_" + add + ".dat"):
        print('no file')
        return []
    data_concatenated = np.loadtxt(path + "/spike_recorder_" + add + ".dat")
    if data_concatenated.size < 5:
        print('empty file')
        return []
    data_raw = data_concatenated[np.argsort(data_concatenated[:, 1])]
    idx_time = ((data_raw[:, 1] >= begin) * (data_raw[:, 1] <= end))
    data_tmp = data_raw[idx_time]
    idx_id = ((data_tmp[:, 0] >= gids[0]) * (data_tmp[:, 0] <= gids[1]))
    return data_tmp[idx_id]


def compute_synchronize_R(spikes, id_max, resolution):
    """
    Borges et al. 2017 : tools for synchronize during complex phase
    :param spikes: the time of each spike by neurons
    :param id_max: the id of neurons with enough spike and the number of spike by neurons
    :param resolution: the resolution of the simulation
    :return:
    """
    max_nb_spike = np.max(np.array(id_max)[:, 1])
    spikes_time = np.array(spikes)[np.array(id_max)[:, 0]]
    time_spike = np.zeros((len(spikes_time), max_nb_spike + 1))
    for i, spikes in enumerate(spikes_time):
        time_spike[i, 0:len(np.unique(spikes))] = np.unique(spikes)
    list_R = []
    list_mean_phase = []
    current_time_spike = np.array([time_spike[:, 0], time_spike[:, 1] - time_spike[:, 0], time_spike[:, 1]])
    id_current = np.zeros_like(time_spike[:, 0]).astype(int)
    init = np.max(time_spike[:, 0])
    not_time = current_time_spike[2, :] < init
    while np.sum(not_time) != 0:
        id_current[not_time] += 1
        current_time_spike[0, not_time] = current_time_spike[2, not_time]
        current_time_spike[1, not_time] = time_spike[not_time, id_current[not_time] + 1] - time_spike[
            not_time, id_current[not_time]]
        current_time_spike[2, not_time] = time_spike[not_time, id_current[not_time] + 1]
        not_time = current_time_spike[2, :] < init
    end = np.max(time_spike)
    t_end = init
    for t in np.arange(init, end, resolution):
        ki_value = 2 * np.pi * (t - current_time_spike[0, :]) / (current_time_spike[1, :])
        R = np.abs(np.mean(np.exp(ki_value * 1j)))
        omega = np.angle(np.mean(np.exp(ki_value * 1j)))
        if omega < 0.0:
            omega += 2 * np.pi
        list_mean_phase.append(omega)
        list_R.append(R)
        id_update = np.where(current_time_spike[2, :] <= t + resolution)
        while len(id_update[0]) != 0:
            id_current[id_update] += 1
            if 0. in time_spike[id_update, id_current[id_update] + 1]:
                t_end = t
                break
            current_time_spike[0, id_update] = time_spike[id_update, id_current[id_update]]
            current_time_spike[1, id_update] = time_spike[id_update, id_current[id_update] + 1] - time_spike[
                id_update, id_current[id_update]]
            current_time_spike[2, id_update] = time_spike[id_update, id_current[id_update] + 1]
            id_update = np.where(current_time_spike[2, :] < t + resolution)
        if t_end != init:
            break
    welch_phase = welch_psd(list_mean_phase, fs=1. / (resolution * 1.e-3))
    frequency_phase = [welch_phase[0][np.argmax(welch_phase[1])], welch_phase[1][np.argmax(welch_phase[1])]]
    return [list_R, [init, t_end], frequency_phase]


def detection_burst(spike_i, limit_burst):
    """
    detection of burst : approximation of burst by a threshold of the inter-spike interval
    :param spike_i: spike train
    :param limit_burst: threshold of the time for checking if it's a burst or not
    :return:
    """
    isis = isi(spike_i)
    isis_add = np.insert(isis, 0, limit_burst + 10.0)
    isis_add = np.append(isis_add, limit_burst + 10.0)
    id_begin_burst = np.where((isis_add > limit_burst) & (np.roll(isis_add, -1) <= limit_burst))[0]
    id_end_burst = np.where((isis_add <= limit_burst) & (np.roll(isis_add, -1) > limit_burst))[0]
    time_start_burst = spike_i[id_begin_burst]
    time_stop_burst = spike_i[id_end_burst]
    nb_burst = id_end_burst - id_begin_burst + 1
    return [isis, time_start_burst, time_stop_burst, nb_burst]


def transfom_spike_global(gid, data, begin, end, resolution, limit_burst):
    """
    Compute different measure on the spike
    :param gid: the array with the id
    :param data: the times spikes
    :param begin: the first time
    :param end: the end time
    :param resolution: the resolution of the simulation
    :param limit_burst: threshold of the time for checking if it's a burst or not
    :return:
        histogram :  hist_0_1, hist_1, hist_5,
        synchronisation : R_list, R_times, percentage,
        inter-spike interval : isi_list
        irregularity : cv_list, lv_list,
        burst analysis : burst_list_nb, burst_list_count, burst_list_rate, burst_list_interval,
            burst_list_begin_cv, burst_list_begin_lv, burst_list_end_cv, burst_list_end_lv,
            percentage_burst, percentage_burst_cv,
        frequency analysis: frequency_hist_0_1, frequency_hist, frequency_phase
    """
    nb_neuron = gid[1] - gid[0]
    if nb_neuron == 0:
        return [None, None, [], [], [], [], [-1, -1], 0.0,
                [], [], [], [],
                [], [], [], [], 0.0, 0.0]
    cv_list = []
    lv_list = []
    isi_list = []
    burst_list_nb = []
    burst_list_count = []
    burst_list_rate = []
    burst_list_interval = []
    burst_list_begin_cv = []
    burst_list_begin_lv = []
    burst_list_end_cv = []
    burst_list_end_lv = []

    spikes = []
    id_spikes_threshold = []
    for i in np.arange(gid[0], gid[1], 1):
        spike_i = data[np.where(data[:, 0] == i)]
        spike_i = (spike_i[np.where(spike_i[:, 1] >= begin), 1]).flatten()
        spike_i = (spike_i[np.where(spike_i <= end)])
        spikes.append(spike_i)
        isis, burst_begin, burst_end, burst_count = detection_burst(spike_i, limit_burst)
        isi_list.append(isis)
        if burst_begin != []:
            burst_list_nb.append(len(burst_begin))
            burst_list_count.append(burst_count)
            burst_list_rate.append(float(len(burst_begin)) * 1000. / float(end - begin))
            burst_list_interval.append(burst_end - burst_begin)
            isis_begin = isi(burst_begin)
            isis_end = isi(burst_end)
            if np.shape(burst_begin)[0] > 20:
                burst_list_begin_cv.append(cv(isis_begin))
                burst_list_begin_lv.append(lv(isis_begin))
                burst_list_end_cv.append(cv(isis_end))
                burst_list_end_lv.append(lv(isis_end))
        if np.shape(spike_i)[0] > 20:  # Filter neurons with enough spikes
            cv_list.append(cv(isis))
            lv_list.append(lv(isis))  # spike in the same time give nan
            id_spikes_threshold.append([len(spikes) - 1, len(spike_i)])
    if id_spikes_threshold:
        R_list, R_times, frequency_phase = compute_synchronize_R(spikes, id_spikes_threshold, resolution)
    else:
        R_list = []
        R_times = [-1, -1]
        frequency_phase = [-1, -1]
    percentage = np.count_nonzero(cv_list) / float(nb_neuron)
    percentage_burst = float(len(burst_list_rate)) / float(nb_neuron)
    percentage_burst_cv = float(len(burst_list_begin_cv)) / float(nb_neuron)
    spikes_concat = np.concatenate(spikes)
    # this suppose the end and begin is in ms
    if int(end - begin) > 1:
        hist_0_1 = np.histogram(spikes_concat, bins=int((end - begin) * 10))  # for bins at 1 milisecond
        welch_hist_0_1 = welch_psd(hist_0_1[0], fs=1. / 1.e-4)
        frequency_hist_0_1 = [welch_hist_0_1[0][np.argmax(welch_hist_0_1[1])],
                              welch_hist_0_1[1][np.argmax(welch_hist_0_1[1])]]
    else:
        hist_0_1 = None
        frequency_hist_0_1 = [-1, -1]
    if int(end - begin) > 1:
        hist_1 = np.histogram(spikes_concat, bins=int(end - begin))  # for bins at 1 milisecond
        welch_hist_1 = welch_psd(hist_1[0], fs=1. / 1.e-3)
        frequency_hist = [welch_hist_1[0][np.argmax(welch_hist_1[1])], welch_hist_1[1][np.argmax(welch_hist_1[1])]]
    else:
        hist_1 = None
        frequency_hist = [-1, -1]

    return [hist_0_1, hist_1, isi_list, cv_list, lv_list, R_list, R_times, percentage,
            burst_list_nb, burst_list_count, burst_list_rate, burst_list_interval,
            burst_list_begin_cv, burst_list_begin_lv, burst_list_end_cv, burst_list_end_lv,
            percentage_burst, percentage_burst_cv,
            frequency_hist_0_1, frequency_hist, frequency_phase]


def compute_rate(result_global, gids, data, begin, end):
    """
    Compute the firing rate
    :param result_global: object for saving result
    :param gids: an array with the range of id for the different population
    :param data: the spike of all neurons between end and begin
    :param begin: the time of the first spike
    :param end: the time of the last spike
    :return: the mean and the standard deviation of firing rate, the maximum and minimum of firing rate
    """
    all_rate = []
    # get data
    n_fil = data[:, 0]
    n_fil = n_fil.astype(int)
    # count the number of the same id
    count_of_n = np.bincount(n_fil)
    count_of_n_fil = count_of_n[gids[0]:gids[1]]
    # compute the rate
    rate_each_n_incomplet = count_of_n_fil * 1000. / (end - begin)
    # fill the table with the neurons which are not firing
    rate_each_n = np.concatenate(
        (
            rate_each_n_incomplet, np.zeros(-np.shape(rate_each_n_incomplet)[0] - gids[0] + gids[1])))
    # save the value
    all_rate = np.concatenate((all_rate, rate_each_n))
    result_global.save_rate(all_rate)
    return 0


def slidding_window(data, width):
    """
    use for mean field
    :param data: instantaneous firing rate
    :param width: windows or times average of the mean field
    :return: state variable of the mean field
    """
    res = np.zeros((data.shape[0] - width, width))
    res[:, :] = np.squeeze(data[np.array([[i + j for i in range(width)] for j in range(data.shape[0] - width)])])
    return res.mean(axis=1)


def compute_irregularity_synchronization(result_global, gids, data, begin, end, resolution,
                                         limit_burst):
    """
        Compute the irregularity and the synchronisation of neurons
        :param result_global where to save the output
        :param gids: an array with the range of id for the different population
        :param data: the spike of all neurons between end and begin
        :param begin: the time of the first spike
        :param end: the time of the last spike
        :param resolution: the resolution of the simulation
        :param limit_burst: threshold of the time for checking if it's a burst or not
        :return: Irregularity : Mean and standard deviation of cv and lv ISI
                 Regularity : Cv of the histogram for a bin of 1 ms and 3 ms
                 Percentage of neurons analyse for the Irregularity]
    """
    # Synchronization and irregularity
    hist_0_1, hist_1, isi_list, cv_list, lv_list, R_list, R_times, percentage, \
    burst_list_nb, burst_list_count, burst_list_rate, burst_list_interval, \
    burst_list_begin_cv, burst_list_begin_lv, burst_list_end_cv, burst_list_end_lv, \
    percentage_burst, percentage_burst_cv, \
    frequency_hist_0_1, frequency_hist, frequency_phase \
        = transfom_spike_global(gids, data, begin, end, resolution, limit_burst)

    # hist
    if hist_1 is None:
        hist_1_variation = None
    else:
        hist_1_variation = variation(hist_1[0])

    # hist
    if hist_0_1 is None:
        hist_0_1_variation = None
    else:
        hist_0_1_variation = variation(hist_0_1[0])

    result_global.save_simple_synchronization(hist_0_1_variation,
                                              hist_1_variation,
                                              )

    # Inter-spike interval
    result_global.save_ISI(isi_list)

    # cv,lv
    result_global.save_irregularity(cv_list, lv_list)

    # R
    result_global.save_R_synchronization(R_list, R_times[0], R_times[1])

    # percentage for Cv, Lv and R
    result_global.save_percentage(percentage)

    # save frequency
    result_global.save_frequency_hist_1(frequency_hist)
    result_global.save_frequency_phase(frequency_phase)

    # Burst
    result_global.save_burst_nb(burst_list_nb)
    result_global.save_burst_count(burst_list_count)
    result_global.save_burst_rate(burst_list_rate)
    result_global.save_burst_interval(burst_list_interval)
    result_global.save_burst_begin_irregularity(burst_list_begin_cv, burst_list_begin_lv)
    result_global.save_burst_end_irregularity(burst_list_end_cv, burst_list_end_lv)
    result_global.save_burst_percentage(percentage_burst)
    result_global.save_burst_percentage_cv(percentage_burst_cv)
    return 0


def analysis_global(path, number, begin, end, resolution, limit_burst):
    """
    The function for global analysis of spikes
    :param path: the path to the file for "population_GIDs.dat" and "spike_detector.gdf"
    :param number: the position of the gid population
    :param begin: the first time
    :param end: the last time
    :param resolution: the resolution of the simulation
    :param limit_burst: threshold of the time for checking if it's a burst or not
    :return: The different measure on the spike
    """
    gids = get_gids(path, number)
    data_all = load_spike(gids, path, begin, end, number)
    result_global = Result_analyse()
    if len(data_all) == 0:
        result_global.empty()
        result_global.set_name_population([gids[2]])
        return result_global.result()
    result_global.save_name_population(gids[2])
    compute_rate(result_global, gids, data_all, begin, end)
    compute_irregularity_synchronization(result_global, gids, data_all, begin, end, resolution, limit_burst)

    # result_global.print_result()

    return result_global.result()


if __name__ == "__main__":
    frequency = 40.0
    analysis_global(
        "/home/kusch/Documents/project/Zerlaut/compare_zerlaut/parameter_analyse/static/simulation/simulation/_frequency_" + str(
            frequency) + "_amplitude_400.0", 0, 0.0, 20000.0, 0.1, 10)
    result_1 = analysis_global(
        "/home/kusch/Documents/project/Zerlaut/compare_zerlaut/parameter_analyse/static/simulation/short/_b_0.0_rate_10.0/", 0, 0.0, 2000.0, 0.1, 10)
    result_2 = analysis_global(
        "/home/kusch/Documents/project/Zerlaut/compare_zerlaut/parameter_analyse/static/simulation/short/_b_0.0_rate_80.0/", 0, 0.0, 2000.0, 0.1, 10)
    print(result_1, result_2)
