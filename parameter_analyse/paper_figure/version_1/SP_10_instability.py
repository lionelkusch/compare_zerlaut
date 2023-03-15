import os
import matplotlib.pyplot as plt
import numpy as np
import parameter_analyse.zerlaut_oscilation.python_file.run.tools_simulation as tools

# parameters for plotting data
labelticks_size = 12
label_legend_size = 12
ticks_size = 12
range_rate = np.array([0.0, 0.2, 0.3, 0.4, 0.6, 1.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0])
frequency = 0.0
# get range of values
result_range = []
for rate in range_rate:
    path = os.path.dirname(__file__) + '/../../zerlaut_oscilation/simulation/deterministe/instability/'
    path_simulation = path + "/rate_" + str(rate) + "/frequency_" + str(frequency)
    result = tools.get_result(path_simulation, 0.0, 2000.0)
    times = result[0][0]
    rateE = result[0][1][:, 0, 0] * 1e3
    stdE = result[0][1][:, 2, 0] * 1e6
    rateI = result[0][1][:, 1, 0] * 1e3
    stdI = result[0][1][:, 4, 0] * 1e6
    corrEI = result[0][1][:, 3, 0]  * 1e6
    adaptationE = result[0][1][:, 5, 0]
    adaptationI = result[0][1][:, 6, 0]
    noise = result[0][1][:, 7, 0]
    result_range.append([times, rateE, rateI, stdE, stdI, corrEI, adaptationE, adaptationI, noise])

# plot data
result_range = np.concatenate([result_range])
fig, axs = plt.subplots(2, 2, figsize=(6.8, 5.5))
plt.subplot(121)
plt.plot(range(len(range_rate)), result_range[:, 1, -1], '.')
plt.xticks(np.array(range(len(range_rate)))[::3], range_rate.astype('str')[::3])
plt.tick_params(labelsize=ticks_size)
plt.xlabel('external input (Hz)', {"fontsize": labelticks_size})
plt.ylabel('firing rate of\nexcitatory population (Hz)', {"fontsize": labelticks_size})
plt.annotate('A', xy=(-0.1, 0.94), xycoords='axes fraction', weight='bold', fontsize=labelticks_size)
axs[0, 1].plot(result_range[:, 0, :].T, result_range[:, 1, :].T)
axs[0, 1].set_ylim(ymax=194.0, ymin=192.0)
axs[0, 1].tick_params(labelsize=ticks_size)
axs[0, 1].annotate('B', xy=(-0.1, 0.90), xycoords='axes fraction', weight='bold', fontsize=labelticks_size)
axs[1, 1].plot(result_range[:, 0, :].T, result_range[:, 1, :].T)
axs[1, 1].set_ylim(ymax=3e-2, ymin=-1e-2)
axs[1, 1].tick_params(labelsize=ticks_size)
axs[1, 1].set_xlabel('time (ms)', {"fontsize": labelticks_size})
axs[1, 1].annotate('C', xy=(-0.1, 0.90), xycoords='axes fraction', weight='bold', fontsize=labelticks_size)
# plt.suptitle('Fix point of zero activities firing rate', fontsize=labelticks_size)

plt.subplots_adjust(top=0.98, bottom=0.09, left=0.13, right=0.98, hspace=0.125, wspace=0.225)
# plt.show()
plt.savefig('./figure/SP_figure_10.png', dpi=300)
