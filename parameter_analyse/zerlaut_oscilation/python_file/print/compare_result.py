import json
import os
import numpy as np
from scipy import signal
import parameter_analyse.zerlaut_oscilation.python_file.run.tools_simulation as tools
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def slidding_window(data, width):
    """
    use for mean field
    :param data: instantaneous firing rate
    :param width: windows or times average of the mean field
    :return: state variable of the mean field
    """
    res = np.zeros((data.shape[0] - width, width))
    res[:, :] = np.squeeze(data[np.array([[i + j for i in range(width)] for j in range(data.shape[0] - width)])])
    return res.mean(axis=1)


frequencies = np.concatenate(([1], np.arange(5., 51., 5.)))
nb_neurons = 10 ** 4
ratio_inhibitory = 0.2
nb_excitatory = int(nb_neurons * (1 - ratio_inhibitory))
nb_inbitory = int(nb_neurons * ratio_inhibitory)
rate = 7.0
dt = 0.1
resolution = 1.0
window_size = 5.0
zoom_frequency = 250
linewidth = 0.5
path_result_mean_field = os.path.dirname(os.path.realpath(__file__)) + '/../../../zerlaut_oscilation/simulation/'
path_result_network = os.path.dirname(os.path.realpath(__file__)) + '/../../../spike_oscilation/simulation/simulation/'

file = PdfPages(os.path.dirname(os.path.realpath(__file__)) + '/../../../compare_rate_' + str(rate) + '.pdf')
for frequency in frequencies:
    path_simulation_mean_field_1 = path_result_mean_field + '/deterministe/rate_' + str(rate) + '/frequency_' + str(
        frequency)
    path_simulation_mean_field_2 = path_result_mean_field + '/stochastic_1e-09/rate_' + str(rate) + '/frequency_' + str(
        frequency)
    path_simulation_mean_field_3 = path_result_mean_field + '/stochastic_1e-08/rate_' + str(rate) + '/frequency_' + str(
        frequency)
    with open(path_simulation_mean_field_1 + '/parameter.json') as f:
        parameters_1 = json.load(f)
    with open(path_simulation_mean_field_2 + '/parameter.json') as f:
        parameters_2 = json.load(f)
    with open(path_simulation_mean_field_3 + '/parameter.json') as f:
        parameters_3 = json.load(f)

    amplitudes = parameters_1['parameter_stimulation']["amp"]
    assert amplitudes == parameters_2['parameter_stimulation']["amp"]
    assert amplitudes == parameters_3['parameter_stimulation']["amp"]
    amplitudes = np.array(amplitudes) * 1e3
    for amplitude_index, amplitude in enumerate(amplitudes):
        for begin, end in [(0.0, 20000.0 - 0.1), (0.0, 1000.0), (19000.0, 20000.0 - 0.1)]:
            print('_frequency_' + str(frequency) + '_amplitude_' + str(amplitude))
            amplitude = amplitudes[amplitude_index]
            path_simulation_network = path_result_network + '/rate_' + str(rate) + '/_frequency_' + str(
                frequency) + '_amplitude_' + str(np.around(amplitude, 2)) + '/'

            # get data
            spikes_concat_ex = np.loadtxt(path_simulation_network + "/spike_recorder_ex.dat")[:, 1]
            spikes_concat_ex = spikes_concat_ex[
                np.where(np.logical_and(spikes_concat_ex >= begin, spikes_concat_ex <= end))]
            spikes_concat_ex = np.concatenate(([begin], spikes_concat_ex, [end]))
            histogram_0_1s_ex = np.array(np.histogram(spikes_concat_ex, bins=int((end - begin) / resolution)),
                                         dtype=object)
            histogram_0_1s_ex[0][0] -= 1
            histogram_0_1s_ex[0][-1] -= 1
            histogram_0_1s_ex[0] = histogram_0_1s_ex[0] / nb_excitatory * 1e3
            histogram_5w_ex = slidding_window(histogram_0_1s_ex[0], int(window_size / resolution))
            spikes_concat_in = np.loadtxt(path_simulation_network + "/spike_recorder_in.dat")
            if spikes_concat_in.shape[0] == 0:
                spikes_concat_in = np.array([])
                pass
            elif len(spikes_concat_in.shape) == 1:
                spikes_concat_in = [spikes_concat_in[1]]
            else:
                spikes_concat_in = spikes_concat_in[:, 1]
                spikes_concat_in = spikes_concat_in[
                    np.where(np.logical_and(spikes_concat_in >= begin, spikes_concat_in <= end))]
            spikes_concat_in = np.concatenate(([begin], spikes_concat_in, [end]))
            histogram_0_1s_in = np.array(np.histogram(spikes_concat_in, bins=int((end - begin) / resolution)),
                                         dtype=object)
            histogram_0_1s_in[0][0] -= 1
            histogram_0_1s_in[0][-1] -= 1
            histogram_0_1s_in[0] = histogram_0_1s_in[0] / nb_inbitory * 1e3
            histogram_5w_in = slidding_window(histogram_0_1s_in[0], int(window_size / resolution))

            result_mean_field_1 = tools.get_result(path_simulation_mean_field_1, begin, end)
            M_times_1 = result_mean_field_1[0][0]
            M_rateE_1 = result_mean_field_1[0][1][:, 0, amplitude_index] * 1e3
            M_stdE_1 = result_mean_field_1[0][1][:, 2, amplitude_index]
            M_rateI_1 = result_mean_field_1[0][1][:, 1, amplitude_index] * 1e3
            M_stdI_1 = result_mean_field_1[0][1][:, 4, amplitude_index]
            M_corrEI_1 = result_mean_field_1[0][1][:, 3, amplitude_index]
            M_adaptationE_1 = result_mean_field_1[0][1][:, 5, amplitude_index]
            result_mean_field_2 = tools.get_result(path_simulation_mean_field_2, begin, end)
            M_times_2 = result_mean_field_2[0][0]
            M_rateE_2 = result_mean_field_2[0][1][:, 0, amplitude_index] * 1e3
            M_stdE_2 = result_mean_field_2[0][1][:, 2, amplitude_index]
            M_rateI_2 = result_mean_field_2[0][1][:, 1, amplitude_index] * 1e3
            M_stdI_2 = result_mean_field_2[0][1][:, 4, amplitude_index]
            M_corrEI_2 = result_mean_field_2[0][1][:, 3, amplitude_index]
            M_adaptationE_2 = result_mean_field_2[0][1][:, 5, amplitude_index]
            result_mean_field_3 = tools.get_result(path_simulation_mean_field_3, begin, end)
            M_times_3 = result_mean_field_3[0][0]
            M_rateE_3 = result_mean_field_3[0][1][:, 0, amplitude_index] * 1e3
            M_stdE_3 = result_mean_field_3[0][1][:, 2, amplitude_index]
            M_rateI_3 = result_mean_field_3[0][1][:, 1, amplitude_index] * 1e3
            M_stdI_3 = result_mean_field_3[0][1][:, 4, amplitude_index]
            M_corrEI_3 = result_mean_field_3[0][1][:, 3, amplitude_index]
            M_adaptationE_3 = result_mean_field_3[0][1][:, 5, amplitude_index]

            plt.figure(figsize=(20, 20))
            plt.suptitle(' Input frequency : ' + str(frequency) + ' amplitude :' + str(amplitude) + ' begin :' + str(
                begin) + ' end:' + str(end))
            ax = plt.subplot(5, 3, 1)
            ax.plot(histogram_5w_ex, linewidth=linewidth)
            ax.plot(histogram_5w_in, linewidth=linewidth)
            ax.set_xlabel('ms')
            ax.set_ylabel('mean spike/' + str(window_size) + 'ms')
            ax = plt.subplot(5, 3, 2)
            f_ex, Pxx_den_ex = signal.welch(histogram_5w_ex, fs=1 / resolution * 1e3,
                                            nperseg=int((end - begin) / resolution) - int(window_size / resolution))
            ax.semilogy(f_ex, Pxx_den_ex)
            f_in, Pxx_den_in = signal.welch(histogram_5w_in, fs=1 / resolution * 1e3,
                                            nperseg=int((end - begin) / resolution) - int(window_size / resolution))
            ax.semilogy(f_in, Pxx_den_in)
            ax.set_title(
                'average mean of the histogram with resolution of ' + str(resolution) + ' ms and a window of 5ms')
            ax.set_xlabel('frequency')
            ax.set_ylabel('power spectrum')
            ax = plt.subplot(5, 3, 3)
            ax.semilogy(f_ex[:zoom_frequency], Pxx_den_ex[:zoom_frequency])
            ax.semilogy(f_in[:zoom_frequency], Pxx_den_in[:zoom_frequency])
            ax.set_xlabel('frequency')
            ax.set_ylabel('power spectrum')

            ax = plt.subplot(5, 3, 4)
            ax.plot(histogram_0_1s_ex[0], linewidth=linewidth * 0.5)
            ax.plot(histogram_0_1s_in[0], linewidth=linewidth * 0.5)
            ax.set_xlabel('ms')
            ax.set_ylabel('nb spike/' + str(resolution) + 'ms')
            ax = plt.subplot(5, 3, 5)
            f_ex, Pxx_den_ex = signal.welch(histogram_0_1s_ex[0], fs=1 / resolution * 1e3,
                                            nperseg=int((end - begin) / resolution) - 1)
            ax.semilogy(f_ex, Pxx_den_ex)
            f_in, Pxx_den_in = signal.welch(histogram_0_1s_in[0], fs=1 / resolution * 1e3,
                                            nperseg=int((end - begin) / resolution) - 1)
            ax.semilogy(f_in, Pxx_den_in)
            ax.set_title('histogram with resolution of ' + str(resolution) + ' ms')
            ax.set_xlabel('frequency')
            ax.set_ylabel('power spectrum')
            ax = plt.subplot(5, 3, 6)
            ax.semilogy(f_ex[:zoom_frequency], Pxx_den_ex[:zoom_frequency])
            ax.semilogy(f_in[:zoom_frequency], Pxx_den_in[:zoom_frequency])
            ax.set_xlabel('frequency')
            ax.set_ylabel('power spectrum')

            ax = plt.subplot(5, 3, 7)
            ax.plot(M_rateE_1, linewidth=linewidth)
            ax.plot(M_rateI_1, linewidth=linewidth)
            ax.set_xlabel('ms')
            ax.set_ylabel('mean firing rate')
            ax = plt.subplot(5, 3, 8)
            f_ex, Pxx_den_ex = signal.welch(M_rateE_1, fs=1 / dt * 1e3, nperseg=int((end - begin - 1) / dt) - 1)
            ax.semilogy(f_ex, Pxx_den_ex)
            f_in, Pxx_den_in = signal.welch(M_rateI_1, fs=1 / dt * 1e3, nperseg=int((end - begin - 1) / dt) - 1)
            ax.semilogy(f_in, Pxx_den_in)
            ax.set_title('Mean field with no noise')
            ax.set_xlabel('frequency')
            ax.set_ylabel('power spectrum')
            ax = plt.subplot(5, 3, 9)
            ax.semilogy(f_ex[:zoom_frequency], Pxx_den_ex[:zoom_frequency])
            ax.semilogy(f_in[:zoom_frequency], Pxx_den_in[:zoom_frequency])
            ax.set_xlabel('frequency')
            ax.set_ylabel('power spectrum')

            ax = plt.subplot(5, 3, 10)
            ax.plot(M_rateE_2, linewidth=linewidth)
            ax.plot(M_rateI_2, linewidth=linewidth)
            ax.set_xlabel('ms')
            ax.set_ylabel('mean firing rate')
            ax = plt.subplot(5, 3, 11)
            f_ex, Pxx_den_ex = signal.welch(M_rateE_2, fs=1 / dt * 1e3, nperseg=int((end - begin) / dt) - 1)
            ax.semilogy(f_ex, Pxx_den_ex)
            f_in, Pxx_den_in = signal.welch(M_rateI_2, fs=1 / dt * 1e3, nperseg=int((end - begin) / dt) - 1)
            ax.semilogy(f_in, Pxx_den_in)
            ax.set_title('Mean field with noise = 1e-9')
            ax.set_xlabel('frequency')
            ax.set_ylabel('power spectrum')
            ax = plt.subplot(5, 3, 12)
            ax.semilogy(f_ex[:zoom_frequency], Pxx_den_ex[:zoom_frequency])
            ax.semilogy(f_in[:zoom_frequency], Pxx_den_in[:zoom_frequency])
            ax.set_xlabel('frequency')
            ax.set_ylabel('power spectrum')

            ax = plt.subplot(5, 3, 13)
            ax.plot(M_rateE_3, linewidth=linewidth)
            ax.plot(M_rateI_3, linewidth=linewidth)
            ax.set_xlabel('ms')
            ax.set_ylabel('mean firing rate')
            ax = plt.subplot(5, 3, 14)
            f_ex, Pxx_den_ex = signal.welch(M_rateE_3, fs=1 / dt * 1e3, nperseg=int((end - begin) / dt) - 1)
            ax.semilogy(f_ex, Pxx_den_ex)
            f_in, Pxx_den_in = signal.welch(M_rateI_3, fs=1 / dt * 1e3, nperseg=int((end - begin) / dt) - 1)
            ax.semilogy(f_in, Pxx_den_in)
            ax.set_title('Mean field with noise = 1e-8')
            ax.set_xlabel('frequency')
            ax.set_ylabel('power spectrum')
            ax = plt.subplot(5, 3, 15)
            ax.semilogy(f_ex[:zoom_frequency], Pxx_den_ex[:zoom_frequency])
            ax.semilogy(f_in[:zoom_frequency], Pxx_den_in[:zoom_frequency])
            ax.set_xlabel('frequency')
            ax.set_ylabel('power spectrum')

            plt.subplots_adjust(hspace=0.6, bottom=0.05, top=0.93, left=0.05, right=0.97)
            plt.savefig(file, format='pdf')
            plt.close('all')
file.close()
