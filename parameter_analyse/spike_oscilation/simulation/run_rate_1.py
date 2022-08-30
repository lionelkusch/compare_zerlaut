from parameter_analyse.spike_oscilation.python_file.run.run_exploration import run_exploration_2D
from parameter_analyse.spike_oscilation.python_file.parameters import parameter_default
import numpy as np

path='/home/kusch/Documents/project/Zerlaut/compare_zerlaut/parameter_analyse/spike_oscilation/simulation/simulation_rate_amplitude/'

frequencies = np.concatenate(([1], np.arange(5., 51., 10.)))
amplitude = np.arange(0.5, 7., 0.5)
parameter_default.param_nest['local_num_threads'] = 8
parameter_default.param_background['rate_equals_amplitude'] = True
run_exploration_2D(path, parameter_default, path+'amplitude_frequency.db', 'first_exploration',
                   {'frequency':frequencies, 'amplitude':amplitude},
                   0.0, 20000.0, analyse=True, simulation=True)
