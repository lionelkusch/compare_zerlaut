#  Copyright 2023 Aix-Marseille Université
# "Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements; and to You under the Apache License, Version 2.0. "
import os
import numpy as np
from parameter_analyse.spike_oscilation.python_file.run.run_exploration import run_exploration_2D, analysis
from parameter_analyse.spike_oscilation.python_file.parameters import parameter_default

frequencies = np.around(np.concatenate(([1], np.arange(5., 51., 5.))), decimals=1)
amplitude = np.around(np.concatenate((np.arange(1.5, 7., 0.5), np.arange(0.1, 1.5, 0.1))), decimals=1)

path = os.path.dirname(os.path.realpath(__file__)) + '/simulation/rate_0.0/'
parameter_default.param_nest['local_num_threads'] = 8
parameter_default.param_topology['excitatory_param']['b'] = 0.0
parameter_default.param_background['rate'] = 0.0
parameter_default.param_background['rate_equals_amplitude'] = False
# comments for testing analysis
# analysis(path + '_frequency_1.0_amplitude_3.0/', parameter_default, path + 'amplitude_frequency_test.db',
#          'first_exploration', {'frequency': 1.0, 'amplitude': 3.0}, 2500.0, 20000.0)
run_exploration_2D(path, parameter_default, path + 'amplitude_frequency.db', 'first_exploration',
                   {'frequency': frequencies, 'amplitude': amplitude},
                   2500.0, 20000.0, analyse=True, simulation=True)

path = os.path.dirname(os.path.realpath(__file__)) + '/simulation/rate_7.0/'
parameter_default.param_nest['local_num_threads'] = 8
parameter_default.param_topology['excitatory_param']['b'] = 0.0
parameter_default.param_background['rate'] = 7.0
parameter_default.param_background['rate_equals_amplitude'] = False
run_exploration_2D(path, parameter_default, path + 'amplitude_frequency.db', 'first_exploration',
                   {'frequency': frequencies, 'amplitude': amplitude},
                   2500.0, 20000.0, analyse=True, simulation=True)

path = os.path.dirname(os.path.realpath(__file__)) + '/simulation/rate_amplitude/'
parameter_default.param_nest['local_num_threads'] = 8
parameter_default.param_topology['excitatory_param']['b'] = 0.0
parameter_default.param_background['rate'] = 0.0
parameter_default.param_background['rate_equals_amplitude'] = True
run_exploration_2D(path, parameter_default, path + 'amplitude_frequency.db', 'first_exploration',
                   {'frequency': frequencies, 'amplitude': amplitude},
                   2500.0, 20000.0, analyse=True, simulation=True)

# for adaptation
path = os.path.dirname(os.path.realpath(__file__)) + '/simulation_b_60/rate_0.0/'
parameter_default.param_nest['local_num_threads'] = 8
parameter_default.param_topology['excitatory_param']['b'] = 60.0
parameter_default.param_background['rate'] = 0.0
parameter_default.param_background['rate_equals_amplitude'] = False
run_exploration_2D(path, parameter_default, path + 'amplitude_frequency.db', 'first_exploration',
                   {'frequency': frequencies, 'amplitude': amplitude},
                   2500.0, 20000.0, analyse=True, simulation=True)

path = os.path.dirname(os.path.realpath(__file__)) + '/simulation_b_60/rate_7.0/'
parameter_default.param_nest['local_num_threads'] = 8
parameter_default.param_topology['excitatory_param']['b'] = 60.0
parameter_default.param_background['rate'] = 7.0
parameter_default.param_background['rate_equals_amplitude'] = False
run_exploration_2D(path, parameter_default, path + 'amplitude_frequency.db', 'first_exploration',
                   {'frequency': frequencies, 'amplitude': amplitude},
                   2500.0, 20000.0, analyse=True, simulation=True)

path = os.path.dirname(os.path.realpath(__file__)) + '/simulation_b_60/rate_amplitude/'
parameter_default.param_nest['local_num_threads'] = 8
parameter_default.param_topology['excitatory_param']['b'] = 60.0
parameter_default.param_background['rate'] = 0.0
parameter_default.param_background['rate_equals_amplitude'] = True
run_exploration_2D(path, parameter_default, path + 'amplitude_frequency.db', 'first_exploration',
                   {'frequency': frequencies, 'amplitude': amplitude},
                   2500.0, 20000.0, analyse=True, simulation=True)

# more
path = os.path.dirname(os.path.realpath(__file__)) + '/simulation/rate_2.5/'
parameter_default.param_nest['local_num_threads'] = 8
parameter_default.param_topology['excitatory_param']['b'] = 0.0
parameter_default.param_background['rate'] = 2.5
parameter_default.param_background['rate_equals_amplitude'] = False
run_exploration_2D(path, parameter_default, path + 'amplitude_frequency.db', 'first_exploration',
                   {'frequency': frequencies, 'amplitude': amplitude},
                   2500.0, 20000.0, analyse=True, simulation=True)

path = os.path.dirname(os.path.realpath(__file__)) + '/simulation_b_60/rate_2.5/'
parameter_default.param_nest['local_num_threads'] = 8
parameter_default.param_topology['excitatory_param']['b'] = 60.0
parameter_default.param_background['rate'] = 2.5
parameter_default.param_background['rate_equals_amplitude'] = False
run_exploration_2D(path, parameter_default, path + 'amplitude_frequency.db', 'first_exploration',
                   {'frequency': frequencies, 'amplitude': amplitude},
                   2000.0, 20000.0, analyse=True, simulation=False)
