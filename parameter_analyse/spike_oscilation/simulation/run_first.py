from parameter_analyse.spike_oscilation.python_file.run.run_exploration import run_exploration_2D
from parameter_analyse.spike_oscilation.python_file.parameters import parameter_default
import numpy as np

path='/home/kusch/Documents/project/Zerlaut/compare_zerlaut/parameter_analyse/spike_oscilation/simulation/simulation/'

frequencies = np.concatenate(([1], np.arange(5., 51., 5.)))
amplitude = np.arange(1., 51., 1.) * 400
parameter_default.param_nest['local_num_threads'] = 8
run_exploration_2D(path, parameter_default, 'amplitude_frequency_2.db', 'first_exploration',
                   {'frequency':frequencies, 'amplitude':amplitude},
                   0.0, 20000.0, analyse=True, simulation=False)