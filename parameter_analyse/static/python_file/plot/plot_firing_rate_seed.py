import os
import numpy as np
import matplotlib.pyplot as plt

from parameter_analyse.spike_oscilation.python_file.print.print_exploration_analysis import getData

table_name = 'first_exploration'
list_variable =[
                {'name': 'b', 'title': 'b', 'min': -1.0, 'max': 90.0},
                {'name': 'rate', 'title': 'rate', 'min': 1.0, 'max': 200.0},
                ]
for b, max_seed in [(0.0, 30), (60.0, 30)]:
    datas = {'excitatory': [], 'inhibitory': []}
    for master_seed in range(max_seed):
        for population in datas.keys():
            data_base = os.path.dirname(os.path.realpath(__file__)) + '/../../simulation/master_seed_' + str(master_seed) + '/database.db'
            data_global = getData(data_base, table_name, list_variable, population, cond=' AND b = '+str(b)+' ')
            datas[population].append(data_global['rates_average'])
    plt.figure()
    plt.plot(data_global['rate'], np.swapaxes(datas['excitatory'], 0, 1), '.')
    plt.figure()
    plt.boxplot(np.concatenate([datas['excitatory']], axis=0), positions=data_global['rate'], meanline=True, showmeans=True)
plt.show()

