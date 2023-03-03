import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.io.matlab import loadmat
from parameter_analyse.analyse_dynamic.print_figure.print_stability import print_stability
from parameter_analyse.static.python_file.plot.helper_function import get_gids_all, load_spike_all, slidding_window
import parameter_analyse.zerlaut_oscilation.python_file.run.tools_simulation as tools

## path of the data
path_init = os.path.dirname(os.path.realpath(__file__)) + "/../../static/simulation/master_seed_0/"
path = os.path.dirname(__file__) + '/../../analyse_dynamic/matlab/'

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
end = 500.0
color = ['red', 'blue']

## load bifurcation
b_0 = loadmat(path + '/EQ_Low/EQ_low.mat', chars_as_strings=True, simplify_cells=True)
b_0['x'][:2] *= 1e3
b_0['x'][-1] *= 1e3
b_0['x'][2:5] *= 1e6

## get network result biffurcation
network_0 = np.concatenate([np.expand_dims(range(100), axis=1), np.load(path_init + '/0.0_mean_var.npy')], axis=1)

## get result_run_80hz
result_10 = tools.get_result(path_init + '/../../../zerlaut_oscilation/simulation/deterministe/short/rate_10.0/frequency_0.0', begin, end)
times_10 = result_10[0][0]
rateE_10 = result_10[0][1][:, 0, :] * 1e3
stdE_10 = result_10[0][1][:, 2, :]
rateI_10 = result_10[0][1][:, 1, :] * 1e3
stdI_10 = result_10[0][1][:, 4, :]
corrEI_10 = result_10[0][1][:, 3, :]
adaptationE_10 = result_10[0][1][:, 5, :]
adaptationI_10 = result_10[0][1][:, 6, :]
noise_10 = result_10[0][1][:, 7, :]
gids_all_10 = get_gids_all(path_init + '/../short/_b_0.0_rate_10.0/')
nb_ex_10 = gids_all_10['excitatory'][0][1] - gids_all_10['excitatory'][0][0]
nb_in_10 = gids_all_10['inhibitory'][0][1] - gids_all_10['inhibitory'][0][0]
data_pop_all_10 = load_spike_all(gids_all_10, path_init + '/../short/_b_0.0_rate_10.0/', begin, end)
hist_ex_10 = np.histogram(data_pop_all_10['excitatory'][1], bins=int((end - begin) / dt))
hist_slide_ex_10 = slidding_window(hist_ex_10[0], int(window / dt)) / nb_ex_10 / (dt * 1e-3)
hist_in_10 = np.histogram(data_pop_all_10['inhibitory'][1], bins=int((end - begin) / dt))
hist_slide_in_10 = slidding_window(hist_in_10[0], int(window / dt)) / nb_in_10 / (dt * 1e-3)

## get result_run_80hz
result_80 = tools.get_result(path_init + '/../../../zerlaut_oscilation/simulation/deterministe/short/rate_80.0/frequency_0.0', begin, end)
times_80 = result_80[0][0]
rateE_80 = result_80[0][1][:, 0, :] * 1e3
stdE_80 = result_80[0][1][:, 2, :]
rateI_80 = result_80[0][1][:, 1, :] * 1e3
stdI_80 = result_80[0][1][:, 4, :]
corrEI_80 = result_80[0][1][:, 3, :]
adaptationE_80 = result_80[0][1][:, 5, :]
adaptationI_80 = result_80[0][1][:, 6, :]
noise_80 = result_80[0][1][:, 7, :]
gids_all_80 = get_gids_all(path_init + '/../short/_b_0.0_rate_80.0/')
nb_ex_80 = gids_all_80['excitatory'][0][1] - gids_all_80['excitatory'][0][0]
nb_in_80 = gids_all_80['inhibitory'][0][1] - gids_all_80['inhibitory'][0][0]
data_pop_all_80 = load_spike_all(gids_all_80, path_init + '/../short/_b_0.0_rate_80.0/', begin, end)
hist_ex_80 = np.histogram(data_pop_all_80['excitatory'][1], bins=int((end - begin) / dt))
hist_slide_ex_80 = slidding_window(hist_ex_80[0], int(window / dt)) / nb_ex_80 / (dt * 1e-3)
hist_in_80 = np.histogram(data_pop_all_80['inhibitory'][1], bins=int((end - begin) / dt))
hist_slide_in_80 = slidding_window(hist_in_80[0], int(window / dt)) / nb_in_80 / (dt * 1e-3)


## make figure
fig = plt.figure(figsize=(6.8, 3.))

## excitatory population
ax = plt.subplot(132)
# ## Trajectory
# ax.plot(hist_slide_ex_10, hist_slide_in_10, linewidth=linewidth, label='network 10Hz')
# ax.plot(rateE_10, rateI_10, linewidth=linewidth, label='mean field 10Hz')
# ax.plot(hist_slide_ex_80, hist_slide_in_80,  linewidth=linewidth, label='network 80Hz')
# ax.plot(rateE_80, rateI_80, linewidth=linewidth, label='mean field 80Hz')
## plot stability of the mean field
line_b_30 = print_stability(b_0['x'], b_0['f'], b_0['s'], 0, 1, color='k', letter=False, linewidth=linewidth_stability)
## plot network mean firing rate
ax.plot(network_0[:, 1], network_0[:, 3], 'xk', ms=marker_size, label='network estimation')
ax.legend(loc='lower right', fontsize=label_legend_size)

ax.scatter([network_0[10, 1], network_0[80, 1]],
           [network_0[10, 3], network_0[80, 3]], c='r', marker='x', s=marker_size_mean, zorder=20)
index_10 = np.argmin(np.abs(b_0['x'][-1]-10) - 100 * (b_0['x'][0] < 100))
index_80 = np.argmin(np.abs(b_0['x'][-1]-80))
ax.scatter([b_0['x'][0][index_10], b_0['x'][0][index_80]],
          [b_0['x'][1][index_10], b_0['x'][1][index_80]], c='r', marker='.', s=marker_size_mean, zorder=20)

## configure the figure
# plt.legend(loc='lower right')
# plt.vlines(0.0, ymin=-30.0, ymax=200.0, color='m')
ax.set_xlim(xmax=200.0, xmin=-0.1)
# plt.xticks([0.0, 50.0, 100.0])
ax.set_xlabel("inhibitory firing rate (Hz)", {"fontsize": labelticks_size})
ax.set_ylim(ymax=200.0, ymin=-0.1)
# plt.yticks([0.0, 100.0, 200.0])
ax.set_ylabel("excitatory firing rate (Hz)", {"fontsize": labelticks_size}, labelpad=0.0)
ax.tick_params(labelsize=ticks_size)
# plt.title('excitatory population', {"fontsize": labelticks_size})




# external input = 10Hz

ax = plt.subplot(234)
for pop, [neurons_id, times_spike] in enumerate(data_pop_all_10.values()):
    ax.plot(times_spike, neurons_id, '.', color=color[pop], markersize=spike_size)
ax.set_xlim(xmax=end + 10.0, xmin=begin - 10.0)
ax.tick_params(axis='both', labelsize=ticks_size)
ax.set_xlabel('time (ms)', {"fontsize": labelticks_size}, labelpad=2.5)
ax.set_ylabel('index neuron', {"fontsize": labelticks_size}, labelpad=-5.0)

ax = plt.subplot(231)
ax.plot(times_10[:-int(window / dt)], hist_slide_ex_10, linewidth=linewidth_network, c='r')
ax.plot(times_10[:-int(window / dt)], hist_slide_in_10, linewidth=linewidth_network, c='b')
ax.plot(times_10, rateE_10, linewidth=linewidth_mean_field, c='orange')
ax.plot(times_10, rateI_10, linewidth=linewidth_mean_field, c='cyan')
ax.tick_params(axis='both', labelsize=ticks_size)
ax.set_xticks([])
ax.set_ylabel('mean firing\nrate (Hz)', {"fontsize": labelticks_size}, labelpad=-1.0)

# external input = 80Hz

ax = plt.subplot(236)
for pop, [neurons_id, times_spike] in enumerate(data_pop_all_80.values()):
    ax.plot(times_spike, neurons_id, '.', color=color[pop], markersize=spike_size)
ax.set_xlim(xmax=end + 10.0, xmin=begin - 10.0)
ax.tick_params(axis='both', labelsize=ticks_size)
ax.set_xlabel('time (ms)', {"fontsize": labelticks_size}, labelpad=2.5)

ax = plt.subplot(233)
ax.plot(times_80[:-int(window / dt)], hist_slide_ex_80, linewidth=linewidth_network, c='r')
ax.plot(times_80[:-int(window / dt)], hist_slide_in_80, linewidth=linewidth_network, c='b')
ax.plot(times_80, rateE_80, linewidth=linewidth_mean_field, c='orange')
ax.plot(times_80, rateI_80, linewidth=linewidth_mean_field, c='cyan')
ax.tick_params(axis='both', labelsize=ticks_size)
ax.set_xticks([])


plt.subplots_adjust(top=0.98, bottom=0.150, left=0.095, right=0.995, wspace=0.33, hspace=0.045)

# plt.show()
plt.savefig('./figure/figure_0.png', dpi=300)
