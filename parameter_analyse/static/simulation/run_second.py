#  Copyright 2023 Aix-Marseille Université
# "Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements; and to You under the Apache License, Version 2.0. "
import os
import datetime
from parameter_analyse.static.python_file.parameters import parameter_default
from parameter_analyse.static.python_file.run.run_exploration import save_parameter, generate_parameter
from parameter_analyse.static.python_file.simulation.simulation_time_evolve import simulate
from parameter_analyse.static.python_file.plot.helper_function import get_gids_all, load_spike_all_long
from parameter_analyse.static.python_file.analysis.analysis_global import time_scale
import numpy as np


def run_sim(results_path, parameter_default, dict_variable, duration, max_step, extra=0):
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
    simulate(results_path=results_path, duration=duration,
             param_nest=param_nest, param_topology=param_topology,
             param_connexion=param_connexion, param_background=param_background,
             max_step=max_step, extra=extra
             )

    print('time: ' + str(datetime.datetime.now()) + ' END SIMULATION \n')


def get_autocorrelation_time(path_init,
                             firing_rate_ext_init=17.0, firing_rate_end=20.0, increment_firing_rate=1.0,
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
    data_pop_all = load_spike_all_long(path_init, firing_rate_ext_init, firing_rate_end, increment_firing_rate)

    for index_fr, firing_rate in enumerate(np.arange(firing_rate_ext_init, firing_rate_end, increment_firing_rate)):
        name_firing_rate = str(np.around(firing_rate))
        firing_rates.append([firing_rate, data_pop_all[name_firing_rate]['excitatory'].shape[0] / nb_ex / (interval_time * 1e-3),
                             data_pop_all[name_firing_rate]['inhibitory'].shape[0] / nb_in / (interval_time * 1e-3)])
        for spikes_concat in [data_pop_all[name_firing_rate]['excitatory'][:, 1], data_pop_all[name_firing_rate]['inhibitory'][:, 1]]:
            hist_0_1 = np.histogram(spikes_concat, bins=int(interval_time * 10))
            hist_0_1_cc_hist_tmp, hist_0_1_lags_tmp, hist_0_1_timescale_tmp = time_scale(hist_0_1, dt=0.1, duration=interval_time)
            hist_0_1_cc_hist.append(hist_0_1_cc_hist_tmp)
            hist_0_1_lags.append(hist_0_1_lags_tmp)
            hist_0_1_timescale.append(hist_0_1_timescale_tmp)
            hist_1 = np.histogram(spikes_concat, bins=int(interval_time))
            hist_1_cc_hist_tmp, hist_1_lags_tmp, hist_1_timescale_tmp = time_scale(hist_1, dt=1., duration=interval_time)
            hist_1_cc_hist.append(hist_1_cc_hist_tmp)
            hist_1_lags.append(hist_1_lags_tmp)
            hist_1_timescale.append(hist_1_timescale_tmp)

    np.save(path_init + '/result_1.npy', np.array([['input', 'firing_rate', 'hist_0_1_cc_hist',
                                                  'hist_0_1_lags', 'hist_0_1_timescale', 'hist_1_cc_hist', 'hist_1_lags', 'hist_1_timescale'],
                                                 np.arange(firing_rate_ext_init, firing_rate_end, increment_firing_rate),
                                                   firing_rates, hist_0_1_cc_hist,
                                                   hist_0_1_lags, hist_0_1_timescale, hist_1_cc_hist,
                                                   hist_1_lags, hist_1_timescale]))

if __name__ == '__main__':
    for b in [0.0, 30., 60.]:
        path = os.path.dirname(os.path.realpath(__file__)) + '/data/time_reduce/b_'+str(b)+'/'
        parameter_default.param_nest['local_num_threads'] = 8
        parameter_default.param_topology['mean_w_0'] = 200.0 if b != 0.0 else 0.0
        # run_sim(path, parameter_default, {'b': b, 'rate': 52.0}, 10000.0, 53, extra=0)
        get_autocorrelation_time(path, firing_rate_ext_init=52.0, firing_rate_end=-0.1, increment_firing_rate=-1.0,
                                 interval_time=10000.0)
