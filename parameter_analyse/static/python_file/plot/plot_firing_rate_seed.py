#  Copyright 2023 Aix-Marseille Université
# "Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements; and to You under the Apache License, Version 2.0. "
import os
import numpy as np
import matplotlib.pyplot as plt

from parameter_analyse.spike_oscilation.python_file.print.print_exploration_analysis import getData

table_name = 'first_exploration'
list_variable = [
                {'name': 'b', 'title': 'b', 'min': -1.0, 'max': 90.0},
                {'name': 'rate', 'title': 'rate', 'min': 1.0, 'max': 200.0},
                ]
for b, max_seed in [(0.0, 30), (60.0, 30)]:
    datas = {'excitatory': {'rate':[]}, 'inhibitory': {'rate':[]}}
    for master_seed in range(max_seed):
        for population in datas.keys():
            data_base = os.path.dirname(os.path.realpath(__file__)) + '/../../simulation/data/master_seed_' \
                        + str(master_seed) + '/database.db'
            data_global = getData(data_base, table_name, list_variable, population, cond=' AND b = '+str(b)+' ')
            datas[population]['rate'].append(data_global['rates_average'])
    plt.figure()
    plt.violinplot(np.concatenate([datas['excitatory']['rate']], axis=0), positions=data_global['rate'], showmeans=True,
                   widths=1.0, showmedians=False,
                   showextrema=False, points=1000, bw_method=0.3)
    plt.figure()
    plt.violinplot(np.concatenate([datas['inhibitory']['rate']], axis=0), positions=data_global['rate'], showmeans=True,
                   widths=1.0, showmedians=False,
                   showextrema=False, points=1000, bw_method=0.3)

plt.show()

