#  Copyright 2023 Aix-Marseille Université
# "Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements; and to You under the Apache License, Version 2.0. "
import os
import datetime
import numpy as np
from parameter_analyse.static.python_file.parameters import parameter_default
from parameter_analyse.static.python_file.run.run_exploration import save_parameter, generate_parameter
from parameter_analyse.static.python_file.simulation.simulation_time_evolve import simulate_2
from parameter_analyse.static.python_file.plot.helper_function import get_gids_all
from elephant.spike_train_correlation import spike_train_timescale, cross_correlation_histogram
from elephant.conversion import BinnedSpikeTrain
import quantities as pq
from parameter_analyse.static.python_file.analysis.analysis_global import slidding_window


def run_sim(results_path, parameter_default, dict_variable, duration, max_step_1, max_step_2, extra=0):
    """
    Run one simulation, analyse the simulation and save this result in the database
    simulation where the external firing reduce each step of specific duration
    :param results_path: the folder where to save spikes
    :param parameter_default: default parameters for simulation
    :param dict_variable : dictionary with the variable change
    :param duration: duration of each step
    :param max_step: maximum of step
    :param extra: extra step
    :return: nothing
    """
    print('time: ' + str(datetime.datetime.now()) + ' BEGIN SIMULATION \n')
    # create the folder for result is not exist
    newpath = os.path.join(os.getcwd(), results_path)
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    elif os.path.exists(newpath + '/spike_recorder_ex.dat'):
        print('Simulation already done ')
        print('time: ' + str(datetime.datetime.now()) + ' END SIMULATION \n')
        return

    param_nest, param_topology, param_connexion, param_background = generate_parameter(parameter_default, dict_variable)

    save_parameter({"param_nest": param_nest, "param_topology": param_topology,
                    "param_connexion": param_connexion, "param_background": param_background},
                   results_path)

    # simulate
    simulate_2(results_path=results_path, duration=duration,
             param_nest=param_nest, param_topology=param_topology,
             param_connexion=param_connexion, param_background=param_background,
             max_step_1=max_step_1, shift_1=1.0,
             max_step_2=max_step_2, shift_2=-1.0, extra=extra
             )

    print('time: ' + str(datetime.datetime.now()) + ' END SIMULATION \n')


def load_spike_all_long_2(path, firing_rate_init, firing_rate_min, increment_firing_rate, index):
    """
    Get the id of the neurons which create the spike
    :param path: path of the file
    :param firing_rate_init: start of firing rate
    :param firing_rate_min: minimum of firing rate
    :param increment_firing_rate: value of reducing firing rate
    :return: The spike of all neurons between end and begin
    """
    data = {}
    for firing_rate in np.arange(firing_rate_init, firing_rate_min, increment_firing_rate):
        print(firing_rate)
        name_firing_rate = str(np.around(firing_rate))
        data[name_firing_rate] = {'excitatory': [], 'inhibitory': []}
        for name, short_name in [('excitatory', 'ex'), ('inhibitory', 'in')]:
            data_concatenated = np.loadtxt(path + "/"+index+name_firing_rate+"_spike_recorder_"+short_name+".dat")
            data_raw = data_concatenated[np.argsort(data_concatenated[:, 1])]
            data[name_firing_rate][name] = data_raw
    return data


def get_firing_rate(path_init,
                    firing_rate_ext_init_1=17.0, firing_rate_end_1=20.0, increment_firing_rate_1=1.0,
                    firing_rate_ext_init_2=20.0, firing_rate_end_2=17.0, increment_firing_rate_2=-1.0,
                    interval_time=10000.0
                    ):
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
    firing_rates = []
    data_pop_all = load_spike_all_long_2(path_init, firing_rate_ext_init_1, firing_rate_end_1, increment_firing_rate_1, "_1_")
    for firing_rate in np.arange(firing_rate_ext_init_1, firing_rate_end_1, increment_firing_rate_1):
        name_firing_rate = str(np.around(firing_rate))
        firing_rates.append([firing_rate, data_pop_all[name_firing_rate]['excitatory'].shape[0] / nb_ex / (interval_time * 1e-3),
                             data_pop_all[name_firing_rate]['inhibitory'].shape[0] / nb_in / (interval_time * 1e-3)])
    data_pop_all = load_spike_all_long_2(path_init, firing_rate_ext_init_2, firing_rate_end_2, increment_firing_rate_2, "_2_")
    for firing_rate in np.arange(firing_rate_ext_init_2, firing_rate_end_2, increment_firing_rate_2):
        name_firing_rate = str(np.around(firing_rate))
        firing_rates.append([firing_rate, data_pop_all[name_firing_rate]['excitatory'].shape[0] / nb_ex / (interval_time * 1e-3),
                             data_pop_all[name_firing_rate]['inhibitory'].shape[0] / nb_in / (interval_time * 1e-3)])
    np.save(path_init + '/firing_rate.npy', firing_rates)


def get_autocorrelation_time(path_init,
                             firing_rate_ext_init_1=17.0, firing_rate_end_1=20.0, increment_firing_rate_1=1.0,
                             firing_rate_ext_init_2=20.0, firing_rate_end_2=17.0, increment_firing_rate_2=-1.0,
                             interval_time=10000.0, lag=50
                            ):
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
    firing_rates = []
    hist_0_1_cc_hist = []
    hist_0_1_lags = []
    hist_0_1_timescale = []
    hist_1_cc_hist = []
    hist_1_lags = []
    hist_1_timescale = []
    hist_5_cc_hist = []
    hist_5_lags = []
    hist_5_timescale = []
    hist_w5_cc_hist = []
    hist_w5_lags = []
    hist_w5_timescale = []

    for data_pop_all, firing_rate_ext_init, firing_rate_end, increment_firing_rate in [[load_spike_all_long_2(path_init, firing_rate_ext_init_1, firing_rate_end_1, increment_firing_rate_1, "_1_"), firing_rate_ext_init_1, firing_rate_end_1, increment_firing_rate_1],
                         [load_spike_all_long_2(path_init, firing_rate_ext_init_2, firing_rate_end_2, increment_firing_rate_2, "_2_"), firing_rate_ext_init_2, firing_rate_end_2, increment_firing_rate_2]
                        ]:
        for index_fr, firing_rate in enumerate(np.arange(firing_rate_ext_init, firing_rate_end, increment_firing_rate)):
            name_firing_rate = str(np.around(firing_rate))
            firing_rates.append([firing_rate, data_pop_all[name_firing_rate]['excitatory'].shape[0] / nb_ex / (interval_time * 1e-3),
                                 data_pop_all[name_firing_rate]['inhibitory'].shape[0] / nb_in / (interval_time * 1e-3)])
            for spikes_concat in [data_pop_all[name_firing_rate]['excitatory'][:, 1], data_pop_all[name_firing_rate]['inhibitory'][:, 1]]:
                hist_0_1 = np.histogram(spikes_concat, bins=int(interval_time * 10))
                hist_0_1_bin_hist = BinnedSpikeTrain(np.expand_dims(hist_0_1[0], 0), t_start=0 * pq.ms,
                                                     t_stop=interval_time * pq.ms, bin_size=0.1 * pq.ms)
                hist_0_1_cc_hist_tmp, hist_0_1_lags_tmp = cross_correlation_histogram(hist_0_1_bin_hist, hist_0_1_bin_hist,
                                                                              window=[-lag, lag],
                                                                              cross_correlation_coefficient=True)
                hist_0_1_cc_hist.append(hist_0_1_cc_hist_tmp)
                hist_0_1_lags.append(hist_0_1_lags_tmp)
                hist_0_1_timescale.append(spike_train_timescale(hist_0_1_bin_hist, max_tau=lag * pq.ms))
                hist_1 = np.histogram(spikes_concat, bins=int(interval_time))
                hist_1_bin_hist = BinnedSpikeTrain(np.expand_dims(hist_1[0], 0), t_start=0 * pq.ms,
                                                     t_stop=interval_time * pq.ms, bin_size=1. * pq.ms)
                hist_1_cc_hist_tmp, hist_1_lags_tmp = cross_correlation_histogram(hist_1_bin_hist, hist_1_bin_hist,
                                                                                      window=[-lag, lag],
                                                                                      cross_correlation_coefficient=True)
                hist_1_cc_hist.append(hist_1_cc_hist_tmp)
                hist_1_lags.append(hist_1_lags_tmp)
                hist_1_timescale.append(spike_train_timescale(hist_1_bin_hist, max_tau=lag * pq.ms))
                hist_5 = np.histogram(spikes_concat, bins=int(interval_time/5))
                hist_5_bin_hist = BinnedSpikeTrain(np.expand_dims(hist_5[0], 0), t_start=0 * pq.ms,
                                                   t_stop=interval_time * pq.ms, bin_size=5 * pq.ms)
                hist_5_cc_hist_tmp, hist_5_lags_tmp = cross_correlation_histogram(hist_5_bin_hist, hist_5_bin_hist,
                                                                                  window=[-lag, lag],
                                                                                  cross_correlation_coefficient=True)
                hist_5_cc_hist.append(hist_5_cc_hist_tmp)
                hist_5_lags.append(hist_5_lags_tmp)
                hist_5_timescale.append(spike_train_timescale(hist_5_bin_hist, max_tau=int(round(lag / 5)) * 5 * pq.ms))
                hist_w5 = slidding_window(hist_0_1[0], 50)
                hist_w5_bin_hist = BinnedSpikeTrain(np.expand_dims(hist_w5, 0), t_start=0 * pq.ms,
                                                   t_stop=(interval_time-5) * pq.ms, bin_size=0.1 * pq.ms)
                hist_w5_cc_hist_tmp, hist_w5_lags_tmp = cross_correlation_histogram(hist_w5_bin_hist, hist_w5_bin_hist,
                                                                                  window=[-lag, lag],
                                                                                  cross_correlation_coefficient=True)
                hist_w5_cc_hist.append(hist_w5_cc_hist_tmp)
                hist_w5_lags.append(hist_w5_lags_tmp)
                hist_w5_timescale.append(spike_train_timescale(hist_w5_bin_hist, max_tau=lag * pq.ms))

    np.save(path_init + '/result.npy', np.array([['input', 'firing_rate', 'hist_0_1_cc_hist',
    'hist_0_1_lags', 'hist_0_1_timescale', 'hist_1_cc_hist', 'hist_1_lags', 'hist_1_timescale', 'hist_5_cc_hist',
    'hist_5_lags', 'hist_5_timescale', 'hist_w5_cc_hist', 'hist_w5_lags', 'hist_w5_timescale'],
    np.concatenate([np.arange(firing_rate_ext_init_1, firing_rate_end_1, increment_firing_rate_1), np.arange(firing_rate_ext_init_2, firing_rate_end_2, increment_firing_rate_2)]),
    firing_rates, hist_0_1_cc_hist,
    hist_0_1_lags, hist_0_1_timescale, hist_1_cc_hist, hist_1_lags, hist_1_timescale, hist_5_cc_hist,
    hist_5_lags, hist_5_timescale, hist_w5_cc_hist, hist_w5_lags, hist_w5_timescale]))


if __name__ == '__main__':
    for b in [30.]:
        path = os.path.dirname(os.path.realpath(__file__)) + '/data/time_reduce_hist/b_'+str(b)+'/'
        parameter_default.param_nest['local_num_threads'] = 8
        parameter_default.param_topology['mean_w_0'] = 200.0 if b != 0.0 else 0.0
        # run_sim(path, parameter_default, {'b': b, 'rate': 17.0}, 10000.0, 60, 60, extra=0)
        # get_firing_rate(path, firing_rate_ext_init_1=17.0, firing_rate_end_1=77.0, increment_firing_rate_1=1.0,
        #                       firing_rate_ext_init_2=77.0, firing_rate_end_2=17.0, increment_firing_rate_2=-1.0,
        #                       interval_time=10000.0)
        get_autocorrelation_time(path, firing_rate_ext_init_1=17.0, firing_rate_end_1=77.0, increment_firing_rate_1=1.0,
                        firing_rate_ext_init_2=77.0, firing_rate_end_2=17.0, increment_firing_rate_2=-1.0,
                        interval_time=10000.0)

