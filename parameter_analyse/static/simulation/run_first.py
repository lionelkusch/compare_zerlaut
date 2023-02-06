import  os
import numpy as np
from parameter_analyse.static.python_file.run.run_exploration import run_exploration_2D
from parameter_analyse.static.python_file.parameters import parameter_default

for master_seed in np.concatenate(([1],range(8, 100))):
    path = os.path.dirname(os.path.realpath(__file__)) + '/master_seed_' + str(master_seed) + '/'
    parameter_default.param_nest['local_num_threads'] = 8
    parameter_default.param_nest['master_seed'] = master_seed
    run_exploration_2D(path, parameter_default, path + 'database.db', 'first_exploration',
                       {'b': np.concatenate([[0.]]), 'rate': np.array(range(100), dtype=float)},
                       1000.0, 5000.0, analyse=True, simulation=True)

# # long simulation
# path = os.path.dirname(os.path.realpath(__file__)) + '/long/'
# parameter_default.param_nest['local_num_threads'] = 8
# parameter_default.param_nest['master_seed'] = 0
# run_exploration_2D(path, parameter_default, path + 'database.db', 'first_exploration',
#                    {'b': np.concatenate([[0., 30., 60.]]), 'rate': [10.0, 50.0, 60.0]},
#                    10000.0, 40000.0, analyse=True, simulation=True)
