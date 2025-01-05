#  Copyright 2023 Aix-Marseille Université
# "Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements; and to You under the Apache License, Version 2.0. "
import os
import numpy as np
import matplotlib.pyplot as plt
from parameter_analyse.zerlaut_oscilation.python_file.print.print_result import getData, grid, draw_point

## path of the data
path_init = os.path.dirname(os.path.realpath(__file__)) + "/../../../static/simulation/data/time_reduce"

## parameter of the figures
labelticks_size = 12
label_legend_size = 12
ticks_size = 12
linewidth = 1.0
marker_size = 3.0

list_variable = [{'name': 'b', 'title': 'b', 'min': 0.0, 'max': 5000000.0},
                 {'name': 'rate', 'title': 'rate', 'min': 0.0, 'max': 5000000.0},
                 ]

for title, path_add in [('high', ''), ('bi', '_hist'), ('low', '_low')]:
    for b in [0.0, 30.0, 60.0]:
        if title != 'bi' or ( b != 0.0 and b != 60.):
            ## get network result
            file = path_init + path_add +'/b_'+str(b)+'/result_1.npy'
            data = np.load(file, allow_pickle=True)

            rates_average = np.array(data[2])[:, 1]
            select_index = np.where(rates_average<80)
            time_scale_0_1 = np.array(data[5])[select_index]*0.1
            time_scale_1 = np.array(data[8])[select_index]
            mean_rate = np.array(data[2])[:, 1][select_index]
            rate_input = np.array(data[1])[select_index]

            plt.figure(figsize=(10, 10))
            plt.suptitle('b = '+ str(b) + str(' ') + title)
            plt.subplot(221)
            plt.plot(rate_input, time_scale_0_1)
            plt.xlabel('input rate (Hz)')
            plt.ylabel('timescale for bin=0.1ms (ms)')
            if len(time_scale_0_1) > 0 and np.max(time_scale_0_1) > 100.0:
                plt.ylim(ymax=50.0)
            plt.ylim(ymin=0.0)
            plt.subplot(222)
            plt.plot(rate_input, mean_rate)
            plt.xlabel('input rate (Hz)')
            plt.ylabel('timescale for bin=1ms (ms)')
            plt.subplot(223)
            plt.plot(rate_input, time_scale_1)
            plt.xlabel('input rate (Hz)')
            plt.ylabel('mean rate (ms)')
            if len(time_scale_1) > 0 and np.max(time_scale_1 ) > 100.0:
                plt.ylim(ymax=50.0)
            plt.ylim(ymin=0.0)
            # plt.subplots_adjust()




plt.show()
