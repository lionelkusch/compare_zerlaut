#  Copyright 2023 Aix-Marseille Université
# "Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements; and to You under the Apache License, Version 2.0. "
import os
import numpy as np
import matplotlib.pyplot as plt
from parameter_analyse.zerlaut_oscilation.python_file.print.print_result import getData, grid, draw_point

## path of the data
path_init = os.path.dirname(os.path.realpath(__file__)) + "/../../../static/simulation/data/master_seed_0/"

## parameter of the figures
labelticks_size = 12
label_legend_size = 12
ticks_size = 12
linewidth = 1.0
marker_size = 3.0

table_name_network = 'first_exploration'
table_name_mean = 'exploration'
population = 'excitatory'

list_variable = [{'name': 'b', 'title': 'b', 'min': 0.0, 'max': 5000000.0},
                 {'name': 'rate', 'title': 'rate', 'min': 0.0, 'max': 5000000.0},
                 ]

## get network result
data_base_network = path_init + '/database_2.db'
data_network = getData(data_base_network, table_name_network, list_variable, population)

rates_average = np.array(data_network['rates_average'])
rates_average[rates_average == None] = np.NAN

for b in [0.0, 30.0, 60.0]:
    select_index = np.where(np.logical_and(np.array(data_network['b']) == b,
                                           rates_average <= 100))
    time_scale_0_1 = np.array(data_network['timescale_0_1ms'])[select_index]*0.1
    time_scale_1 = np.array(data_network['timescale_1ms'])[select_index]
    time_scale_1[np.where(time_scale_1 == -1)] = np.NAN
    mean_rate = np.array(data_network['rates_average'])[select_index]
    rate_input = np.array(data_network['rate'])[select_index]
    plt.figure(figsize=(10, 10))
    plt.suptitle('b = '+ str(b))
    plt.subplot(221)
    plt.plot(rate_input, time_scale_0_1)
    plt.xlabel('input rate (Hz)')
    plt.ylabel('timescale for bin=0.1ms (ms)')
    plt.subplot(222)
    plt.plot(rate_input, mean_rate)
    plt.xlabel('input rate (Hz)')
    plt.ylabel('timescale for bin=1ms (ms)')
    plt.subplot(223)
    plt.plot(rate_input, time_scale_1)
    plt.xlabel('input rate (Hz)')
    plt.ylabel('mean rate (ms)')
    plt.subplots_adjust(top=0.95, bottom=0.05, left=0.09, right=0.96)
    plt.savefig('Figure_1_timescale_b_'+str(b)+'.png')

# plt.show()
