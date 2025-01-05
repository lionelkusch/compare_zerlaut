import os
import parameter_analyse.zerlaut_oscilation.python_file.run.tools_simulation as tools
from parameter_analyse.static.python_file.analysis.analysis_global import get_gids, load_spike, slidding_window
import matplotlib.pyplot as plt
import numpy as np


begin = 0.0
end = 4001.0
dt = 0.1
path_root = os.path.dirname(os.path.realpath(__file__)) + '/../../../'

for rate in [0.0, 7.0]:
    for frequency in [5.0, 20.0]:
        path_simulation_network = path_root + '/spike_oscilation/simulation/simulation/rate_'+str(rate)+'/_frequency_'+str(frequency)+'_amplitude_3.0/'
        duration = end - begin
        gids = get_gids(path_simulation_network, 0) # 0 excitatory
        data_all = load_spike(gids, path_simulation_network, begin, end, 0)
        hist_ex = np.histogram(data_all[:, 1], bins=int((end - begin) / dt))
        hist_ex_1 = np.histogram(data_all[:, 1], bins=int((end - begin)))
        gids = get_gids(path_simulation_network, 1) # 0 inhibitory
        data_all = load_spike(gids, path_simulation_network, begin, end, 1)
        hist_in = np.histogram(data_all[:, 1], bins=int((end - begin) / dt))
        hist_in_1 = np.histogram(data_all[:, 1], bins=int((end - begin)))

        fig_ex = plt.figure(figsize=(10, 8))
        plt.suptitle('excitatory rate:' + str(rate) + ' frequency:' + str(frequency) + ' amplitude:3.0')
        plt.subplot(211)
        ax_ex_high = plt.gca()
        plt.subplot(212)
        ax_ex_low = plt.gca()
        plt.subplots_adjust(hspace=0.05)

        fig_in = plt.figure(figsize=(10, 8))
        plt.suptitle('inhibitory rate:' + str(rate) + ' frequency:' + str(frequency) + ' amplitude:3.0')
        plt.subplot(211)
        ax_in_high = plt.gca()
        plt.subplot(212)
        ax_in_low = plt.gca()
        plt.subplots_adjust(hspace=0.05)

        # result_network = []
        result_T = []
        for T in [0.2, 0.5, 1, 5, 50, 150]:
            print(rate, frequency, T)
            path_simulation_mean_field = path_root +'/zerlaut_oscilation/simulation/deterministe/T_' + str(
            T) + "/b_0.0/rate_" + str(rate) + "/frequency_" + str(frequency)
            result = tools.get_result(path_simulation_mean_field, begin, end)
            times = result[0][0]
            rateE = result[0][1][:, 0, :]
            rateI = result[0][1][:, 1, :]
            w = result[0][1][:, 5, :]
            external_input_excitatory_to_excitatory = result[0][1][:, 8, :]
            external_input_excitatory_to_excitatory[np.where(external_input_excitatory_to_excitatory < 0)] = 0.0
            external_input_excitatory_to_inhibitory = result[0][1][:, 9, :]
            external_input_excitatory_to_inhibitory[np.where(external_input_excitatory_to_inhibitory < 0)] = 0.0
            result_T.append([times, rateE, rateI, external_input_excitatory_to_excitatory, external_input_excitatory_to_inhibitory])
            # result_network.append([slidding_window(hist_ex[0], int(T/dt)), slidding_window(hist_in[0], int(T/dt))])
            ax_ex_high.plot(result_T[-1][0], result_T[-1][1]*1e3, '.', label='T:'+str(T), markersize=2.)
            if T == 5:
                ax_ex_high.plot(result_T[-1][0], result_T[-1][3] * 1e3, '--b', label='stimulation')
                ax_ex_high.plot(np.array(range(len(hist_ex[0])))*dt, hist_ex[0],
                           color='r', linewidth=0.05)
                ax_ex_high.plot(np.array(range(len(slidding_window(hist_ex[0], int(20/dt))))) * dt, slidding_window(hist_ex[0], int(20/dt)),
                           '--', color='r', label='network sliding(20ms)')
            # if T == 0.2:
            #     ax2 = ax_ex_high.twinx()
            #     ax2.plot(result_T[-1][0], w, label='w', color='brown')
            ax_ex_high.legend()
            ax_ex_high.set_ylim(ymin=180.0, ymax=200.0)
            ax_ex_low.plot(result_T[-1][0], result_T[-1][1]*1e3, '.', label='T:'+str(T), markersize=2.)
            if T == 5:
                ax_ex_low.plot(result_T[-1][0], result_T[-1][3] * 1e3, '--b', label='stimulation')
                ax_ex_low.plot(np.array(range(len(hist_ex[0])))*dt, hist_ex[0],
                           color='r', linewidth=0.05)
                ax_ex_low.plot(np.array(range(len(slidding_window(hist_ex[0], int(20/dt))))) * dt, slidding_window(hist_ex[0], int(20/dt)),
                           '--', color='r', label='network sliding \n(20ms)')
            # if T == 0.2:
            #     ax2 = ax_ex_low.twinx()
            #     ax2.plot(result_T[-1][0], w, label='w', color='brown')
            # ax_ex_low.legend()
            ax_ex_low.set_ylim(ymax=50.0, ymin=0.0)
            ax_in_high.plot(result_T[-1][0], result_T[-1][2]*1e3, '.', label='T:'+str(T), markersize=2.)
            if T == 5:
                ax_in_high.plot(result_T[-1][0], result_T[-1][4]*1e3, '--b', label='stimulation')
                ax_in_high.plot(np.array(range(len(hist_in[0])))*dt, hist_in[0],
                           color='r', linewidth=0.05)
                ax_in_high.plot(np.array(range(len(slidding_window(hist_in[0], int(20/dt)))))*dt, slidding_window(hist_in[0], int(20/dt)),
                           '--', color = 'r', label = 'network sliding \n(20ms)')
            ax_in_high.legend()
            ax_in_high.set_ylim(ymin=180.0, ymax=200.0)
            ax_in_low.plot(result_T[-1][0], result_T[-1][2] * 1e3, '.', label='T:' + str(T), markersize=2.)
            if T == 5:
                ax_in_low.plot(result_T[-1][0], result_T[-1][4] * 1e3, '--b', label='stimulation')
                ax_in_low.plot(np.array(range(len(hist_in[0]))) * dt, hist_in[0],
                           color='r', linewidth=0.05)
                ax_in_low.plot(np.array(range(len(slidding_window(hist_in[0], int(20 / dt))))) * dt,
                           slidding_window(hist_in[0], int(20 / dt)),
                           '--', color='r', label='network sliding(20ms)')
            ax_in_low.set_ylim(ymax=50.0, ymin=0.0)
            # ax_in_low.legend()
            # fig_ex.savefig('figure_compare_time_serie_rate_'+str(rate)+'_frequency_'+str(frequency)+'_pop_excitatory.png')
            # fig_in.savefig('figure_compare_time_serie_rate_'+str(rate)+'_frequency_'+str(frequency)+'_pop_inhibitory.png')
            ax_ex_high.set_xlim(xmin=3000.0)
            ax_ex_low.set_xlim(xmin=3000.0)
            ax_in_high.set_xlim(xmin=3000.0)
            ax_in_low.set_xlim(xmin=3000.0)
            fig_ex.savefig('figure_compare_time_serie_rate_'+str(rate)+'_frequency_'+str(frequency)+'_pop_excitatory_zoom.png')
            fig_in.savefig('figure_compare_time_serie_rate_'+str(rate)+'_frequency_'+str(frequency)+'_pop_inhibitory_zoom.png')
        # plt.show()


