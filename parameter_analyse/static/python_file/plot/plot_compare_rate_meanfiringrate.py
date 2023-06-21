#  Copyright 2023 Aix-Marseille Université
# "Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements; and to You under the Apache License, Version 2.0. "
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.matlab import loadmat
from parameter_analyse.spike_oscilation.python_file.print.print_exploration_analysis import getData
from parameter_analyse.analyse_dynamic.print_figure.print_stability import print_stability

path_init = os.path.dirname(os.path.realpath(__file__)) + "/../../simulation/data/master_seed_0/"
path = os.path.dirname(__file__) + '/../../../analyse_dynamic/matlab/'
table_name = 'first_exploration'
list_variable = [
                 {'name': 'b', 'title': 'b', 'min': -1.0, 'max': 90.0},
                 {'name': 'rate', 'title': 'rate', 'min': 1.0, 'max': 200.0}
                ]
for path_bifurcation, b in [('/EQ_Low/EQ_low.mat', 0.0),
                            ('/b_30/EQ_Low/EQ_Low.mat', 30.0),
                            ('/b_60/EQ_Low/EQ_Low.mat', 60.0)]:
    # get network result
    network_data = np.concatenate([np.expand_dims(range(100), axis=1), np.load(path_init + '/'+str(b)+'_mean_var.npy')],
                                  axis=1)
    # bifurcation data:
    bifurcation_data = loadmat(path+path_bifurcation)
    bifurcation_data['x'][:2] *= 1e3
    bifurcation_data['x'][-1] *= 1e3
    bifurcation_data['x'][2:5] *= 1e7
    if b == 0.0:
        bifurcation_data['x'] = np.vstack((bifurcation_data['x'][:5], np.zeros((1, bifurcation_data['x'].shape[1]), ),
                                           bifurcation_data['x'][5]))
    datas = {'excitatory': {'rate': [], 'std': [], 'std_rate':[]}, 'inhibitory': {'rate': [], 'std': [], 'std_rate':[]}}
    for population in datas.keys():
        data_base = path_init + '/database.db'
        data_global = getData(data_base, table_name, list_variable, population, cond=' AND b = '+str(b)+' ')
        datas[population]['rate'].append(data_global['rates_average'])
        datas[population]['std'].append(data_global['rates_std'])
        datas[population]['std_rate'].append(np.array(data_global['cvs_IFR_1ms'])*np.array(data_global['rates_average']))
        datas[population]['std_rate'].append(np.array(data_global['cvs_IFR_0_1ms'])*np.array(data_global['rates_average']))
    plt.figure()
    plt.plot(data_global['rate'], datas['excitatory']['rate'][0], 'x', ms=5.0)
    print_stability(bifurcation_data['x'], bifurcation_data['f'], bifurcation_data['s'], 6, 0, letter=True, linewidth=1.0)
    plt.plot(network_data[:, 0], network_data[:, 1], 'x', ms=5.0)

    plt.figure()
    plt.plot(data_global['rate'], datas['excitatory']['std_rate'][0], 'x', ms=5.0)
    print_stability(bifurcation_data['x'], bifurcation_data['f'], bifurcation_data['s'], 6, 2, letter=True, linewidth=1.0)
    plt.plot(network_data[:, 0], network_data[:, 2], 'x', ms=5.0)
    plt.figure()
    plt.plot(data_global['rate'], datas['inhibitory']['rate'][0], 'x', ms=5.0)
    print_stability(bifurcation_data['x'], bifurcation_data['f'], bifurcation_data['s'], 6, 1, letter=True, linewidth=1.0)
    plt.plot(network_data[:, 0], network_data[:, 3], 'x', ms=5.0)
    plt.figure()
    plt.plot(data_global['rate'], datas['inhibitory']['std_rate'][0], 'x', ms=5.0)
    print_stability(bifurcation_data['x'], bifurcation_data['f'], bifurcation_data['s'], 6, 4, letter=True, linewidth=1.0)
    plt.plot(network_data[:, 0], network_data[:, 4], 'x', ms=5.0)
    plt.show()

