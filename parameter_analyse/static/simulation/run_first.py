#  Copyright 2023 Aix-Marseille Université
# "Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements; and to You under the Apache License, Version 2.0. "
import os
import numpy as np
from parameter_analyse.static.python_file.run.run_exploration import run_exploration_2D
from parameter_analyse.static.python_file.parameters import parameter_default

simulation =False
analysis = True

# first estimation
path = os.path.dirname(os.path.realpath(__file__)) + '/data/master_seed_0/'
parameter_default.param_nest['local_num_threads'] = 8
parameter_default.param_nest['master_seed'] = 0
run_exploration_2D(path, parameter_default, path + 'database_2.db', 'first_exploration',
                   {'b': np.concatenate([[0., 30., 60.]]), 'rate': np.array(range(100), dtype=float)},
                   10000.0, 40000.0, analyse=analysis, simulation=simulation)

# short simulation
path = os.path.dirname(os.path.realpath(__file__)) + '/data/short/'
parameter_default.param_nest['local_num_threads'] = 8
parameter_default.param_nest['master_seed'] = 0
run_exploration_2D(path, parameter_default, path + 'database_2.db', 'first_exploration',
                   {'b': np.concatenate([[0.]]), 'rate': [80.0]},
                   000.0, 2000.0, analyse=analysis, simulation=simulation)

# long simulation
path = os.path.dirname(os.path.realpath(__file__)) + '/data/long/'
parameter_default.param_nest['local_num_threads'] = 8
parameter_default.param_nest['master_seed'] = 0
run_exploration_2D(path, parameter_default, path + 'database_2.db', 'first_exploration',
                   {'b': np.concatenate([[0., 30., 60.]]), 'rate': [10.0, 50.0, 60.0]},
                   10000.0, 40000.0, analyse=analysis, simulation=simulation)

for master_seed in [0]: #range(30):
    master_seed = int(master_seed)
    path = os.path.dirname(os.path.realpath(__file__)) + '/data/master_seed_' + str(master_seed) + '/'
    parameter_default.param_nest['local_num_threads'] = 8
    parameter_default.param_nest['master_seed'] = master_seed
    run_exploration_2D(path, parameter_default, path + 'database_2.db', 'first_exploration',
                       {'b': np.concatenate([[0.]]), 'rate': np.array(range(100), dtype=float)},
                       1000.0, 5000.0, analyse=analysis, simulation=simulation)

for master_seed in [0]:#range(0, 30):
    master_seed = int(master_seed)
    path = os.path.dirname(os.path.realpath(__file__)) + '/data/master_seed_' + str(master_seed) + '/'
    parameter_default.param_nest['local_num_threads'] = 8
    parameter_default.param_nest['master_seed'] = master_seed
    run_exploration_2D(path, parameter_default, path + 'database_2.db', 'first_exploration',
                       {'b': np.concatenate([[30.0, 60.]]), 'rate': np.array(range(100), dtype=float)},
                       1000.0, 5000.0, analyse=analysis, simulation=simulation)


# short simulation
path = os.path.dirname(os.path.realpath(__file__)) + '/data/short_b/'
parameter_default.param_nest['local_num_threads'] = 8
parameter_default.param_nest['master_seed'] = 0
run_exploration_2D(path, parameter_default, path + 'database_2.db', 'first_exploration',
                   {'b': np.concatenate([[0., 30., 60.]]), 'rate': [25., 50.0, 75.]},
                   000.0, 4000.0, analyse=analysis, simulation=simulation)