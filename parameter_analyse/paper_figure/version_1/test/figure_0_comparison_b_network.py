#  Copyright 2023 Aix-Marseille Université
# "Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements; and to You under the Apache License, Version 2.0. "
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.io.matlab import loadmat
from parameter_analyse.analyse_dynamic.print_figure.print_stability import print_stability
from parameter_analyse.static.python_file.plot.helper_function import get_gids_all, load_spike_all, slidding_window
import parameter_analyse.zerlaut_oscilation.python_file.run.tools_simulation as tools

## path of the data
path_init = os.path.dirname(os.path.realpath(__file__))
path = os.path.dirname(__file__) + '/../../../analyse_dynamic/matlab/'

## parameter of the figures
labelticks_size = 10
label_legend_size = 7
ticks_size = 10
linewidth_network = 0.5
linewidth_mean_field = 1.0
linewidth_stability = 0.5
marker_size = 5.0
marker_size_mean = 30.0
spike_size = 0.006

window = 5.0
dt = 0.1
begin = 0.0
end = 2000.0
rate = 75.0
color = ['red', 'blue']

## get network result biffurcation
network_0 = np.concatenate([np.expand_dims(range(100), axis=1), np.load(path_init + '/../../../static/simulation/data/master_seed_0//0.0_mean_var.npy')], axis=1)

# ## get result_run_b_0
result_b_0 = tools.get_result(path_init + '/../../../zerlaut_oscilation/simulation/deterministe/short_b/b_0.0/rate_'+str(rate)+'/frequency_0.0', begin, end)
times_b_0 = result_b_0[0][0]
rateE_b_0 = result_b_0[0][1][:, 0, :] * 1e3
stdE_b_0 = result_b_0[0][1][:, 2, :]
rateI_b_0 = result_b_0[0][1][:, 1, :] * 1e3
stdI_b_0 = result_b_0[0][1][:, 4, :]
corrEI_b_0 = result_b_0[0][1][:, 3, :]
adaptationE_b_0 = result_b_0[0][1][:, 5, :]
adaptationI_b_0 = result_b_0[0][1][:, 6, :]
noise_b_0 = result_b_0[0][1][:, 7, :]

# ## get result_run_b_30
result_b_30 = tools.get_result(path_init + '/../../../zerlaut_oscilation/simulation/deterministe/short_b/b_30.0/rate_'+str(rate)+'/frequency_0.0', begin, end)
times_b_30 = result_b_30[0][0]
rateE_b_30 = result_b_30[0][1][:, 0, :] * 1e3
stdE_b_30 = result_b_30[0][1][:, 2, :]
rateI_b_30 = result_b_30[0][1][:, 1, :] * 1e3
stdI_b_30 = result_b_30[0][1][:, 4, :]
corrEI_b_30 = result_b_30[0][1][:, 3, :]
adaptationE_b_30 = result_b_30[0][1][:, 5, :]
adaptationI_b_30 = result_b_30[0][1][:, 6, :]
noise_b_30 = result_b_30[0][1][:, 7, :]

# ## get result_run_b_60
result_b_60 = tools.get_result(path_init + '/../../../zerlaut_oscilation/simulation/deterministe/short_b/b_60.0/rate_'+str(rate)+'/frequency_0.0', begin, end)
times_b_60 = result_b_60[0][0]
rateE_b_60 = result_b_60[0][1][:, 0, :] * 1e3
stdE_b_60 = result_b_60[0][1][:, 2, :]
rateI_b_60 = result_b_60[0][1][:, 1, :] * 1e3
stdI_b_60 = result_b_60[0][1][:, 4, :]
corrEI_b_60 = result_b_60[0][1][:, 3, :]
adaptationE_b_60 = result_b_60[0][1][:, 5, :]
adaptationI_b_60 = result_b_60[0][1][:, 6, :]
noise_b_60 = result_b_60[0][1][:, 7, :]


for d_b_0, d_b_30, d_b_60 in [(rateE_b_0, rateE_b_30, rateE_b_60),
                             (rateI_b_0, rateI_b_30, rateI_b_60),
                             (stdE_b_0, stdE_b_30, stdE_b_60),
                             (stdI_b_0, stdI_b_30, stdI_b_60),
                             (corrEI_b_0, corrEI_b_30, corrEI_b_60),
                             (adaptationE_b_0, adaptationE_b_30, adaptationE_b_60),
                             (adaptationI_b_0, adaptationI_b_30, adaptationI_b_60),
                             (noise_b_0, noise_b_30, noise_b_60),
                             ]:
    plt.figure()
    plt.plot(times_b_0, d_b_0, linewidth=linewidth_mean_field, c='blue')
    plt.plot(times_b_30, d_b_30, linewidth=linewidth_mean_field, c='red')
    plt.plot(times_b_60, d_b_60, linewidth=linewidth_mean_field, c='orange')

plt.show()