#  Copyright 2023 Aix-Marseille Université
# "Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements; and to You under the Apache License, Version 2.0. "
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.matlab import loadmat
from parameter_analyse.spike_oscilation.python_file.print.print_exploration_analysis import getData
from parameter_analyse.analyse_dynamic.print_figure.print_stability import print_stability


## parameter of the figures
labelticks_size = 12
label_legend_size = 12
ticks_size = 12
linewidth = 1.0
marker_size = 3.0
letters = [['A', 'B'], ['C', 'D']]

# get data
# parameter of the data
path_init = os.path.dirname(os.path.realpath(__file__)) + "/../../static/simulation/data/"
path = os.path.dirname(__file__) + '/../../analyse_dynamic/matlab/'
max_seed = 30
box_plot = False
table_name = 'first_exploration'
list_variable = [
                 {'name': 'b', 'title': 'b', 'min': -1.0, 'max': 90.0},
                 {'name': 'rate', 'title': 'rate', 'min': 1.0, 'max': 200.0}
                ]
datas={}
for path_bifurcation, b in [('/EQ_Low/EQ_low.mat', 0.0),
                            ('/b_60/EQ_Low/EQ_Low.mat', 60.0)]:
    # get data from each seed
    datas[str(b)] = {'excitatory': {'rate': [], 'std': [], 'std_rate':[]}, 'inhibitory': {'rate': [], 'std': [], 'std_rate':[]}}
    for master_seed in range(max_seed):
        for population in datas[str(b)].keys():
            data_base = path_init + '/master_seed_'+str(master_seed)+'/database.db'
            data_global = getData(data_base, table_name, list_variable, population, cond=' AND b = '+str(b)+' ')
            datas[str(b)][population]['rate'].append(data_global['rates_average'])
            datas[str(b)][population]['std'].append(data_global['rates_std'])
            datas[str(b)][population]['std_rate'].append(np.array(data_global['cvs_IFR_1ms'])*np.array(data_global['rates_average']))
            datas[str(b)][population]['std_rate'].append(np.array(data_global['cvs_IFR_0_1ms']) * np.array(data_global['rates_average']))
    # # get network result for seed 0
    # network_data = np.concatenate([np.expand_dims(range(100), axis=1), np.load(path_init + '/'+str(b)+'_mean_var.npy')], axis=1)
    # bifurcation data:
    bifurcation_data = loadmat(path+path_bifurcation)
    bifurcation_data['x'][:2] *= 1e3
    bifurcation_data['x'][-1] *= 1e3
    bifurcation_data['x'][2:5] *= 1e7
    if b == 0.0:
        bifurcation_data['x'] = np.vstack((bifurcation_data['x'][:5], np.zeros((1, bifurcation_data['x'].shape[1]), ), bifurcation_data['x'][5]))
    datas[str(b)]['bifurcation'] = bifurcation_data

# plot figure
fig, axs = plt.subplots(2, 2, figsize=(6.8, 6.8))
for y, (b,color) in enumerate([(0.0,'k'), (60.0,'g')]):
    for x, population in enumerate(['excitatory', 'inhibitory']):
        plt.sca(axs[x, y])
        if box_plot:
            plt.boxplot(np.concatenate([datas[str(b)][population]['rate']], axis=0), positions=data_global['rate'])
        else:
            parts = plt.violinplot(np.concatenate([datas[str(b)][population]['rate']], axis=0), positions=data_global['rate'], showmeans=True, widths=1.0, showmedians=False,
                                 showextrema=False, points=1000, bw_method=0.3)
            for pc in parts['bodies']:
                # pc.set_facecolor('#D43F3A')
                # pc.set_edgecolor('black')
                pc.set_alpha(1.0)
        print_stability(datas[str(b)]['bifurcation']['x'], datas[str(b)]['bifurcation']['f'], datas[str(b)]['bifurcation']['s'], 6, x,  color=color, letter=True, linewidth=1.0)
        plt.xlim(xmin=0.0, xmax=100.0)
        plt.xticks([0.0, 50.0, 100.0], [0, 50, 100])
        plt.yticks([0.0, 100.0, 200.0], [0, 100, 200])
        plt.tick_params(labelsize=ticks_size)
        if x == 1:
            plt.xlabel("external excitatory input (Hz)", {"fontsize": labelticks_size}, labelpad=2.)
            if y == 0:
                plt.ylabel("inhibitory firing rate (Hz)", {"fontsize": labelticks_size}, labelpad=2.)
        if x == 0:
            plt.title("b="+str(b), {"fontsize": labelticks_size})
            if y == 0:
                plt.ylabel("excitatory firing rate (Hz)", {"fontsize": labelticks_size}, labelpad=2.)
        plt.annotate(letters[x][y], xy=(-0.1, 0.9), xycoords='axes fraction', weight='bold', fontsize=labelticks_size)


plt.subplots_adjust(top=0.96, bottom=0.080, left=0.100, right=0.975, wspace=0.17, hspace=0.15)
plt.savefig('./figure/SP_figure_11.png', dpi=300)
# plt.show()

