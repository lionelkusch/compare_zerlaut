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


amplitudes = (np.array(list(range(1, 51, 1))) * 1e-3).tolist()
frequencies = np.concatenate(([1], np.arange(5., 51., 5.)))
noise_1 = 0.0
noise_2 = 1e-9
noise_3 = 1e-8
dt = 0.1
window_size = 5.0
zoom_frequency = 250
linewidth = 0.5
path_result_mean_field = '/home/kusch/Documents/project/Zerlaut/compare_zerlaut/parameter_analyse/zerlaut_oscilation/simulation/'
path_result_network = '/home/kusch/Documents/project/Zerlaut/compare_zerlaut/parameter_analyse/spike_oscilation/simulation/simulation/'

file = PdfPages('/home/kusch/Documents/project/Zerlaut/compare_zerlaut/parameter_analyse/compare.pdf')
for frequency in frequencies:
    for amplitude_index, amplitude in enumerate(amplitudes):
        for begin, end in [(0.0, 20000.0), (19000.0, 20000.0)]:
            print('_frequency_' + str(frequency) + '_amplitude_' + str(amplitude * 400 * 1e3))
            amplitude = amplitudes[amplitude_index]
            path_simulation_network = path_result_network + '/_frequency_' + str(frequency) + '_amplitude_' + str(np.round(amplitude * 400 * 1e3))

            path_simulation_mean_field_1 = path_result_mean_field + '/frequency_' + str(frequency) + '_noise_' + str(noise_1)
            path_simulation_mean_field_2 = path_result_mean_field + '/frequency_' + str(frequency) + '_noise_' + str(noise_2)
            path_simulation_mean_field_3 = path_result_mean_field + '/frequency_' + str(frequency) + '_noise_' + str(noise_3)

            # get data
            spikes_concat_ex = np.loadtxt(path_simulation_network + "/spike_recorder_ex.dat")[:, 1]
            spikes_concat_ex = spikes_concat_ex[np.where(np.logical_and(spikes_concat_ex >= begin, spikes_concat_ex <= end))]
            histogram_0_1s_ex = np.histogram(spikes_concat_ex, bins=int((end - begin) / dt))
            histogram_5w_ex = slidding_window(histogram_0_1s_ex[0], int(window_size / dt))
            spikes_concat_in = np.loadtxt(path_simulation_network + "/spike_recorder_in.dat")[:, 1]
            spikes_concat_in = spikes_concat_in[np.where(np.logical_and(spikes_concat_in >= begin, spikes_concat_in <= end))]
            histogram_0_1s_in = np.histogram(spikes_concat_in, bins=int((end - begin) / dt))
            histogram_5w_in = slidding_window(histogram_0_1s_in[0], int(window_size / dt))

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
            plt.suptitle(' Input frequency : ' + str(frequency) + ' amplitude :' + str(amplitude * 1e3) + ' begin :' + str(
                begin) + ' end:' + str(end))
            ax = plt.subplot(5, 3, 1)
            ax.plot(histogram_5w_ex, linewidth=linewidth)
            ax.plot(histogram_5w_in, linewidth=linewidth)
            ax.set_xlabel('ms')
            ax.set_ylabel('mean spike/5ms')
            ax = plt.subplot(5, 3, 2)
            f_ex, Pxx_den_ex = signal.welch(histogram_5w_ex, fs=1 / dt * 1e3, nperseg=1 / dt * 1e3)
            ax.semilogy(f_ex, Pxx_den_ex)
            f_in, Pxx_den_in = signal.welch(histogram_5w_in, fs=1 / dt * 1e3, nperseg=1 / dt * 1e3)
            ax.semilogy(f_in, Pxx_den_in)
            ax.set_title('average mean of the histogram with resolution of 0.1 ms and a window of 5ms')
            ax.set_xlabel('frequency')
            ax.set_ylabel('power spectrum')
            ax = plt.subplot(5, 3, 3)
            ax.semilogy(f_ex[:zoom_frequency], Pxx_den_ex[:zoom_frequency])
            ax.semilogy(f_in[:zoom_frequency], Pxx_den_in[:zoom_frequency])
            ax.set_xlabel('frequency')
            ax.set_ylabel('power spectrum')

            ax = plt.subplot(5, 3, 4)
            ax.plot(histogram_0_1s_ex[0], linewidth=linewidth)
            ax.plot(histogram_0_1s_in[0], linewidth=linewidth)
            ax.set_xlabel('ms')
            ax.set_ylabel('nb spike/0.1ms')
            ax = plt.subplot(5, 3, 5)
            f_ex, Pxx_den_ex = signal.welch(histogram_0_1s_ex[0], fs=1 / dt * 1e3, nperseg=1 / dt * 1e3)
            ax.semilogy(f_ex, Pxx_den_ex)
            f_in, Pxx_den_in = signal.welch(histogram_0_1s_in[0], fs=1 / dt * 1e3, nperseg=1 / dt * 1e3)
            ax.semilogy(f_in, Pxx_den_in)
            ax.set_title('histogram with resolution of 0.1 ms')
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
            f_ex, Pxx_den_ex = signal.welch(M_rateE_1, fs=1 / dt * 1e3, nperseg=1 / dt * 1e3)
            ax.semilogy(f_ex, Pxx_den_ex)
            f_in, Pxx_den_in = signal.welch(M_rateI_1, fs=1 / dt * 1e3, nperseg=1 / dt * 1e3)
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
            f_ex, Pxx_den_ex = signal.welch(M_rateE_2, fs=1 / dt * 1e3, nperseg=1 / dt * 1e3)
            ax.semilogy(f_ex, Pxx_den_ex)
            f_in, Pxx_den_in = signal.welch(M_rateI_2, fs=1 / dt * 1e3, nperseg=1 / dt * 1e3)
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
            f_ex, Pxx_den_ex = signal.welch(M_rateE_3, fs=1 / dt * 1e3, nperseg=1 / dt * 1e3)
            ax.semilogy(f_ex, Pxx_den_ex)
            f_in, Pxx_den_in = signal.welch(M_rateI_3, fs=1 / dt * 1e3, nperseg=1 / dt * 1e3)
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
