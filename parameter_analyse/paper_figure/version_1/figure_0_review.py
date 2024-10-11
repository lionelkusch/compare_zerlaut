#  Copyright 2023 Aix-Marseille Université
# "Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements; and to You under the Apache License, Version 2.0. "
import os
import matplotlib.pyplot as plt
import numpy as np
from parameter_analyse.static.python_file.plot.helper_function import get_gids_all, load_spike_all, slidding_window
import parameter_analyse.zerlaut_oscilation.python_file.run.tools_simulation as tools


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
color_ex_spike = 'red'#'black'#'green'#'blue'
color_ex_mean = 'orange'#'grey'#'lime'#'cyan'
color_in_spike = 'blue'#'blue'#'red'
color_in_mean = 'cyan'#'cyan'#'orange'



## path of the data
path_init = os.path.dirname(os.path.realpath(__file__))

window = 5.0
dt = 0.1
begin = 0.0
end = 500.0
color = ['red', 'blue']

## get result_run_80hz
gids_all_80 = get_gids_all(path_init + '/../../static/simulation/data/short/_b_0.0_rate_80.0/')
nb_ex_80 = gids_all_80['excitatory'][0][1] - gids_all_80['excitatory'][0][0]
nb_in_80 = gids_all_80['inhibitory'][0][1] - gids_all_80['inhibitory'][0][0]
data_pop_all_80 = load_spike_all(gids_all_80, path_init + '/../../static/simulation/data/short/_b_0.0_rate_80.0/', begin, end)
hist_ex_80 = np.histogram(data_pop_all_80['excitatory'][1], bins=int((end - begin) / dt))
hist_slide_ex_80 = slidding_window(hist_ex_80[0], int(window / dt)) / nb_ex_80 / (dt * 1e-3)
hist_in_80 = np.histogram(data_pop_all_80['inhibitory'][1], bins=int((end - begin) / dt))
hist_slide_in_80 = slidding_window(hist_in_80[0], int(window / dt)) / nb_in_80 / (dt * 1e-3)

T_5_result_80 = tools.get_result(path_init + '/../../zerlaut_oscilation/simulation/deterministe/short/rate_80.0/frequency_0.0', begin, end)
T_5_times_80 = T_5_result_80[0][0]
T_5_rateE_80 = T_5_result_80[0][1][:, 0, :] * 1e3
T_5_stdE_80 = T_5_result_80[0][1][:, 2, :]
T_5_rateI_80 = T_5_result_80[0][1][:, 1, :] * 1e3
T_5_stdI_80 = T_5_result_80[0][1][:, 4, :]
T_5_corrEI_80 = T_5_result_80[0][1][:, 3, :]
T_5_adaptationE_80 = T_5_result_80[0][1][:, 5, :]
T_5_adaptationI_80 = T_5_result_80[0][1][:, 6, :]
T_5_noise_80 = T_5_result_80[0][1][:, 7, :]

T_1_result_80 = tools.get_result(path_init + '/../../zerlaut_oscilation/simulation/deterministe/review/T_1.0/b_0.0/rate_80.0/frequency_0.0', begin, end)
T_1_times_80 = T_1_result_80[0][0]
T_1_rateE_80 = T_1_result_80[0][1][:, 0, :] * 1e3
T_1_stdE_80 = T_1_result_80[0][1][:, 2, :]
T_1_rateI_80 = T_1_result_80[0][1][:, 1, :] * 1e3
T_1_stdI_80 = T_1_result_80[0][1][:, 4, :]
T_1_corrEI_80 = T_1_result_80[0][1][:, 3, :]
T_1_adaptationE_80 = T_1_result_80[0][1][:, 5, :]
T_1_adaptationI_80 = T_1_result_80[0][1][:, 6, :]
T_1_noise_80 = T_1_result_80[0][1][:, 7, :]

T_2_result_80 = tools.get_result(path_init + '/../../zerlaut_oscilation/simulation/deterministe/review/T_2.0/b_0.0/rate_80.0/frequency_0.0', begin, end)
T_2_times_80 = T_2_result_80[0][0]
T_2_rateE_80 = T_2_result_80[0][1][:, 0, :] * 1e3
T_2_stdE_80 = T_2_result_80[0][1][:, 2, :]
T_2_rateI_80 = T_2_result_80[0][1][:, 1, :] * 1e3
T_2_stdI_80 = T_2_result_80[0][1][:, 4, :]
T_2_corrEI_80 = T_2_result_80[0][1][:, 3, :]
T_2_adaptationE_80 = T_2_result_80[0][1][:, 5, :]
T_2_adaptationI_80 = T_2_result_80[0][1][:, 6, :]
T_2_noise_80 = T_2_result_80[0][1][:, 7, :]


plt.figure(figsize=(10, 10))
ax = plt.subplot(221)
ax.plot(T_5_times_80[:-int(window / dt)], hist_slide_ex_80, linewidth=linewidth_network, c=color_ex_spike)
ax.plot(T_5_times_80[:-int(window / dt)], hist_slide_in_80, linewidth=linewidth_network, c=color_in_spike)
ax.plot(T_5_times_80, T_5_rateE_80, linewidth=linewidth_mean_field, c=color_ex_mean)
ax.plot(T_5_times_80, T_5_rateI_80, linewidth=linewidth_mean_field, c=color_in_mean)
ax.tick_params(axis='both', labelsize=ticks_size)
ax.set_xticks([])

ax = plt.subplot(223)
ax.plot(T_5_times_80, T_5_rateE_80, linewidth=linewidth_mean_field, label='T=5')
ax.plot(T_1_times_80, T_1_rateE_80, linewidth=linewidth_mean_field, label='T=1')
ax.plot(T_2_times_80, T_2_rateE_80, linewidth=linewidth_mean_field, label='T=2')
ax.tick_params(axis='both', labelsize=ticks_size)
plt.legend()
plt.title('excitatory')

ax = plt.subplot(224)
ax.plot(T_5_times_80, T_5_rateI_80, linewidth=linewidth_mean_field, label='T=5')
ax.plot(T_1_times_80, T_1_rateI_80, linewidth=linewidth_mean_field, label='T=1')
ax.plot(T_2_times_80, T_2_rateI_80, linewidth=linewidth_mean_field, label='T=2')
plt.legend()
plt.title('inhibitory')
plt.subplots_adjust(top=0.99, left=0.045, right=0.99, bottom=0.04, wspace=0.110)

plt.savefig('figure_0_review.png')
plt.show()