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
path = os.path.dirname(__file__) + '/../../analyse_dynamic/matlab/'

## parameter of the figures
version = '_v1_5'
labelticks_size = 10
label_legend_size = 7
ticks_size = 10
linewidth_mean_field = 1.0
linewidth_stability = 0.5
marker_size_mean = 30.0
marker_size = 5.0
color_ex_spike = 'blue' #'black'#'green'#'blue'
color_ex_mean = 'cyan' #'grey'#'lime'#'cyan'
color_in_spike = 'red'#'blue'#'red'
color_in_mean = 'orange'#'cyan'#'orange'

window = 5.0
dt = 0.1
begin = 0.0
end = 100.0
rate = 50.0
spike_size = 0.01 if rate > 70.0 else 0.01
linewidth_network = 0.05 if rate > 70.0 else 0.5
color = [color_ex_spike, color_in_spike]

## get network result biffurcation
network_0 = np.concatenate([np.expand_dims(range(100), axis=1), np.load(path_init + '/../../static/simulation/data/master_seed_0//0.0_mean_var.npy')], axis=1)

# ## get result_run_b_0
result_b_0 = tools.get_result(path_init + '/../../zerlaut_oscilation/simulation/deterministe/short_b/b_0.0/rate_'+str(rate)+'/frequency_0.0', begin, end)
times_b_0 = result_b_0[0][0]
rateE_b_0 = result_b_0[0][1][:, 0, :] * 1e3
stdE_b_0 = result_b_0[0][1][:, 2, :]
rateI_b_0 = result_b_0[0][1][:, 1, :] * 1e3
stdI_b_0 = result_b_0[0][1][:, 4, :]
corrEI_b_0 = result_b_0[0][1][:, 3, :]
adaptationE_b_0 = result_b_0[0][1][:, 5, :]
adaptationI_b_0 = result_b_0[0][1][:, 6, :]
noise_b_0 = result_b_0[0][1][:, 7, :]
gids_all_b_0 = get_gids_all(path_init + '/../../static/simulation/data/short_b/_b_0.0_rate_'+str(rate)+'/')
nb_ex_b_0 = gids_all_b_0['excitatory'][0][1] - gids_all_b_0['excitatory'][0][0]
nb_in_b_0 = gids_all_b_0['inhibitory'][0][1] - gids_all_b_0['inhibitory'][0][0]
data_pop_all_b_0 = load_spike_all(gids_all_b_0, path_init + '/../../static/simulation/data/short_b/_b_0.0_rate_'+str(rate)+'/', begin, end)
hist_ex_b_0 = np.histogram(data_pop_all_b_0['excitatory'][1], bins=int((end - begin) / dt))
hist_slide_ex_b_0 = slidding_window(hist_ex_b_0[0], int(window / dt)) / nb_ex_b_0 / (dt * 1e-3)
hist_in_b_0 = np.histogram(data_pop_all_b_0['inhibitory'][1], bins=int((end - begin) / dt))
hist_slide_in_b_0 = slidding_window(hist_in_b_0[0], int(window / dt)) / nb_in_b_0 / (dt * 1e-3)

# ## get result_run_b_30
result_b_30 = tools.get_result(path_init + '/../../zerlaut_oscilation/simulation/deterministe/short_b/b_30.0/rate_'+str(rate)+'/frequency_0.0', begin, end)
times_b_30 = result_b_30[0][0]
rateE_b_30 = result_b_30[0][1][:, 0, :] * 1e3
stdE_b_30 = result_b_30[0][1][:, 2, :]
rateI_b_30 = result_b_30[0][1][:, 1, :] * 1e3
stdI_b_30 = result_b_30[0][1][:, 4, :]
corrEI_b_30 = result_b_30[0][1][:, 3, :]
adaptationE_b_30 = result_b_30[0][1][:, 5, :]
adaptationI_b_30 = result_b_30[0][1][:, 6, :]
noise_b_30 = result_b_30[0][1][:, 7, :]
gids_all_b_30 = get_gids_all(path_init + '/../../static/simulation/data/short_b/_b_30.0_rate_'+str(rate)+'/')
nb_ex_b_30 = gids_all_b_30['excitatory'][0][1] - gids_all_b_30['excitatory'][0][0]
nb_in_b_30 = gids_all_b_30['inhibitory'][0][1] - gids_all_b_30['inhibitory'][0][0]
data_pop_all_b_30 = load_spike_all(gids_all_b_30, path_init + '/../../static/simulation/data/short_b/_b_30.0_rate_'+str(rate)+'/', begin, end)
hist_ex_b_30 = np.histogram(data_pop_all_b_30['excitatory'][1], bins=int((end - begin) / dt))
hist_slide_ex_b_30 = slidding_window(hist_ex_b_30[0], int(window / dt)) / nb_ex_b_30 / (dt * 1e-3)
hist_in_b_30 = np.histogram(data_pop_all_b_30['inhibitory'][1], bins=int((end - begin) / dt))
hist_slide_in_b_30 = slidding_window(hist_in_b_30[0], int(window / dt)) / nb_in_b_30 / (dt * 1e-3)

# ## get result_run_b_60
result_b_60 = tools.get_result(path_init + '/../../zerlaut_oscilation/simulation/deterministe/short_b/b_60.0/rate_'+str(rate)+'/frequency_0.0', begin, end)
times_b_60 = result_b_60[0][0]
rateE_b_60 = result_b_60[0][1][:, 0, :] * 1e3
stdE_b_60 = result_b_60[0][1][:, 2, :]
rateI_b_60 = result_b_60[0][1][:, 1, :] * 1e3
stdI_b_60 = result_b_60[0][1][:, 4, :]
corrEI_b_60 = result_b_60[0][1][:, 3, :]
adaptationE_b_60 = result_b_60[0][1][:, 5, :]
adaptationI_b_60 = result_b_60[0][1][:, 6, :]
noise_b_60 = result_b_60[0][1][:, 7, :]
gids_all_b_60 = get_gids_all(path_init + '/../../static/simulation/data/short_b/_b_60.0_rate_'+str(rate)+'/')
nb_ex_b_60 = gids_all_b_60['excitatory'][0][1] - gids_all_b_60['excitatory'][0][0]
nb_in_b_60 = gids_all_b_60['inhibitory'][0][1] - gids_all_b_60['inhibitory'][0][0]
data_pop_all_b_60 = load_spike_all(gids_all_b_60, path_init + '/../../static/simulation/data/short_b/_b_60.0_rate_'+str(rate)+'/', begin, end)
hist_ex_b_60 = np.histogram(data_pop_all_b_60['excitatory'][1], bins=int((end - begin) / dt))
hist_slide_ex_b_60 = slidding_window(hist_ex_b_60[0], int(window / dt)) / nb_ex_b_60 / (dt * 1e-3)
hist_in_b_60 = np.histogram(data_pop_all_b_60['inhibitory'][1], bins=int((end - begin) / dt))
hist_slide_in_b_60 = slidding_window(hist_in_b_60[0], int(window / dt)) / nb_in_b_60 / (dt * 1e-3)

print(hist_slide_ex_b_0[0], hist_slide_in_b_0[0])
print(hist_slide_ex_b_30[0], hist_slide_in_b_30[0])
print(hist_slide_ex_b_60[0], hist_slide_in_b_60[0])

## make figure
fig = plt.figure(figsize=(6.8, 3.), dpi=600)

## excitatory population
ax = plt.subplot(231)
ax.plot(times_b_0[:-int(window / dt)], hist_slide_ex_b_0, linewidth=linewidth_network, c=color_ex_spike)
ax.plot(times_b_0[:-int(window / dt)], hist_slide_in_b_0, linewidth=linewidth_network, c=color_in_spike)
ax.plot(times_b_0, rateE_b_0, linewidth=linewidth_mean_field, c=color_ex_mean)
ax.plot(times_b_0, rateI_b_0, linewidth=linewidth_mean_field, c=color_in_mean)
ax.tick_params(axis='both', labelsize=ticks_size)
ax.set_xticks([])
ax.set_ylabel('mean firing\nrate (Hz)', {"fontsize": labelticks_size}, labelpad=-1.0)
ax.annotate('A', xy=(-0.34, 0.95), xycoords='axes fraction', weight='bold', fontsize=labelticks_size)
ax.set_title('b=0.0pA')

ax = plt.subplot(234)
for pop, [neurons_id, times_spike] in enumerate(data_pop_all_b_0.values()):
    ax.plot(times_spike, neurons_id, ',', color=color[pop], markersize=spike_size)
ax.set_xlim(xmax=end + 10.0, xmin=begin - 10.0)
ax.tick_params(axis='both', labelsize=ticks_size)
ax.set_xlabel('time (ms)', {"fontsize": labelticks_size}, labelpad=2.5)
ax.set_ylabel('index neuron', {"fontsize": labelticks_size}, labelpad=-5.0)
ax.annotate('B', xy=(-0.34, 0.95), xycoords='axes fraction', weight='bold', fontsize=labelticks_size)


# external input = 10Hz
ax = plt.subplot(232)
ax.plot(times_b_30[:-int(window / dt)], hist_slide_ex_b_30, linewidth=linewidth_network, c=color_ex_spike)
ax.plot(times_b_30[:-int(window / dt)], hist_slide_in_b_30, linewidth=linewidth_network, c=color_in_spike)
ax.plot(times_b_30, rateE_b_30, linewidth=linewidth_mean_field, c=color_ex_mean)
ax.plot(times_b_30, rateI_b_30, linewidth=linewidth_mean_field, c=color_in_mean)
ax.tick_params(axis='both', labelsize=ticks_size)
ax.set_xticks([])
ax.set_yticks([])
ax.annotate('C', xy=(-0.08, 0.95), xycoords='axes fraction', weight='bold', fontsize=labelticks_size)
ax.set_title('b=30.0pA')

ax = plt.subplot(235)
for pop, [neurons_id, times_spike] in enumerate(data_pop_all_b_30.values()):
    ax.plot(times_spike, neurons_id, ',', color=color[pop], markersize=spike_size)
ax.set_xlim(xmax=end + 10.0, xmin=begin - 10.0)
ax.tick_params(axis='both', labelsize=ticks_size)
ax.set_xlabel('time (ms)', {"fontsize": labelticks_size}, labelpad=2.5)
ax.set_yticks([])
ax.annotate('D', xy=(-0.08, 0.95), xycoords='axes fraction', weight='bold', fontsize=labelticks_size)


# external input = 80Hz
ax = plt.subplot(233)
ax.plot(times_b_60[:-int(window / dt)], hist_slide_ex_b_60, linewidth=linewidth_network, c=color_ex_spike)
ax.plot(times_b_60[:-int(window / dt)], hist_slide_in_b_60, linewidth=linewidth_network, c=color_in_spike)
ax.plot(times_b_60, rateE_b_60, linewidth=linewidth_mean_field, c=color_ex_mean)
ax.plot(times_b_60, rateI_b_60, linewidth=linewidth_mean_field, c=color_in_mean)
ax.tick_params(axis='both', labelsize=ticks_size)
ax.set_xticks([])
ax.set_yticks([])
ax.annotate('E', xy=(-0.08, 0.95), xycoords='axes fraction', weight='bold', fontsize=labelticks_size)
ax.set_title('b=60.0pA')

ax = plt.subplot(236)
for pop, [neurons_id, times_spike] in enumerate(data_pop_all_b_60.values()):
    ax.plot(times_spike, neurons_id, ',', color=color[pop], markersize=spike_size)
ax.set_xlim(xmax=end + 10.0, xmin=begin - 10.0)
ax.tick_params(axis='both', labelsize=ticks_size)
ax.set_xlabel('time (ms)', {"fontsize": labelticks_size}, labelpad=2.5)
ax.set_yticks([])
ax.annotate('F', xy=(-0.08, 0.95), xycoords='axes fraction', weight='bold', fontsize=labelticks_size)


plt.subplots_adjust(top=0.92, bottom=0.150, left=0.095, right=0.975, wspace=0.115, hspace=0.045)

# plt.show()
plt.savefig('./figure/figure_0_b_rate_'+str(rate)+version+'.png', dpi=600)
plt.savefig('./figure/figure_0_b_rate_'+str(rate)+version+'.tiff', dpi=600)
plt.savefig('./figure/figure_0_b_rate_'+str(rate)+version+'.jpeg', dpi=600)
plt.savefig('./figure/figure_0_b_rate_'+str(rate)+version+'.svg')
plt.savefig('./figure/figure_0_b_rate_'+str(rate)+version+'.pdf')
