import numpy as np
from scipy.signal import butter, lfilter, hilbert
from scipy import signal
from elephant.spectral import welch_psd


def PLV(theta1, theta2):
    """
    calculate PLV from 2 vectors of phases ( Lachaux et al., (1999))
    :param theta1: phase vector of signal 1
    :param theta2: phase vector of signal 2
    :return: Phase Locking Values
    """
    complex_phase_diff = np.exp(1j * (np.unwrap(theta1 - theta2)))
    plv = np.abs(np.sum(complex_phase_diff) / len(theta1))
    angle = np.angle(np.sum(complex_phase_diff) / len(theta1))
    return plv, angle


def compute_PLV_mean_shift(hist, end, frequency, remove_init=500, remove_start_hilbert=3000,
                           remove_end=2500, sampling_rate=1e3, sampling_ms=1.0):
    """
    compute the mean phase shift and the PLV between the input signal and the output
    :param hist: histogram
    :param end: end analysis
    :param frequency: frequency of input
    :param remove_init: remove the beginning
    :param remove_start_hilbert: remove the start of hilbert transform
    :param remove_end: remove the end of hilbert transform
    :param sampling_rate: sampling rate of the histogram
    :param sampling_ms: sampling ms of the histogram
    :return:
    """
    low_frequency = frequency - 0.1
    high_frequency = frequency + 0.1
    I_signal = np.sin(2*np.pi*frequency*np.arange(0.0, end, sampling_ms)*1e-3)
    I_theta = np.angle(hilbert(I_signal))[remove_start_hilbert+remove_init:-remove_end]
    b, a = butter(1, [low_frequency, high_frequency], fs=sampling_rate, btype='band')
    hist_filter = lfilter(b, a, hist[remove_init:])
    theta_filter = np.angle(hilbert(hist_filter))
    hist_filter = hist_filter[remove_start_hilbert:-remove_end]
    theta_filter = theta_filter[remove_start_hilbert:-remove_end]
    PLV_value, PLV_angle = PLV(I_theta, theta_filter)
    mean_phase_shift = np.angle(np.mean(np.cos(I_theta - theta_filter))+1j*np.mean(np.sin(I_theta-theta_filter)))
    return PLV_value, PLV_angle, mean_phase_shift


if __name__ == "__main__":
    import numpy as np

    import parameter_analyse.zerlaut_oscilation.python_file.run.tools_simulation as tools
    from parameter_analyse.zerlaut_oscilation.python_file.parameters.parameter_default import Parameter
    import matplotlib.pyplot as plt

    frequencies = np.concatenate(([1], np.arange(5., 51., 5.)))
    # noises = np.arange(0, 1e-5, 5e-7)
    noises = np.arange(0, 1e-7, 1e-8)
    # noises = np.arange(0, 1e-8, 1e-9)
    amplitudes = (np.arange(1., 51., 1.)*1e-3).tolist()
    for fr in range(2):
        for ns in range(1, 2):
            for amplitude_choice in range(0, 1):
                frequency = frequencies[fr]
                noise = noises[ns]
                print('frequency : '+str(frequency)+' noise : '+str(noise) + ' amplitude: ' + str(amplitudes[amplitude_choice]))
                fs = 1e4
                sampling_ms = 0.1
                begin = 0.0
                end = 20000.0
                path_simulation = '/home/kusch/Documents/project/Zerlaut/compare_zerlaut/parameter_analyse/zerlaut_oscilation/simulation/frequency_'+str(frequency)+'_noise_'+str(noise)+'/'
                # frequency=1.0; noise=5e-8; fs = 1e5; sampling_ms = 0.01; path_simulation = '/home/kusch/Documents/project/Zerlaut/compare_zerlaut/parameter_analyse/zerlaut_oscilation/simulation/aprecise_frequency_'+str(frequency)+'_noise_'+str(noise)+'/'

                result = tools.get_result(path_simulation, begin, end)

                times = result[0][0]
                rateE = result[0][1][:, 0, :]
                stdE = result[0][1][:, 2, :]
                rateI = result[0][1][:, 1, :]
                stdI = result[0][1][:, 4, :]
                corrEI = result[0][1][:, 3, :]
                adaptationE = result[0][1][:, 5, :]

                # Input signal
                I_signal = np.sin(2*np.pi*frequency*np.arange(begin, end, sampling_ms)*1e-3)
                I_input = amplitudes[amplitude_choice] * np.copy(I_signal)
                I_input[np.where(I_input < 0.0)] = 0.0
                I_theta = np.angle(signal.hilbert(I_signal))

                welch_model = welch_psd(rateE[:, amplitude_choice]*1e3, frequency_resolution=1.0, fs=fs, )
                if welch_model[0][np.argmax(welch_model[1])] != frequency:
                    print('Bad max frequency ' + str(welch_model[0][np.argmax(welch_model[1])]) +
                          ' => frequency : ' + str(frequency) + ' noise : ' + str(noise) +
                          ' amplitude: ' + str(amplitudes[amplitude_choice]))
                    # print(welch_model[0][np.argmax(welch_model[1])])
                    # raise Exception('bad frequency')

                # filtering
                b, a = signal.butter(1, [frequency-0.1, frequency+0.1], fs=fs, btype='band')
                hist_filter = signal.lfilter(b, a, rateE[:, amplitude_choice]*1e3)
                theta_filter = np.angle(signal.hilbert(hist_filter))
                theta = np.angle(signal.hilbert(rateE[:, amplitude_choice]*1e3))

                plt.figure()
                ax = plt.subplot(321)
                ax.semilogy(welch_model[0], welch_model[1])
                ax.set_title('frequency : '+str(frequency)+' noise : '+str(noise) + ' amplitude: ' + str(amplitudes[amplitude_choice]))
                ax = plt.subplot(322)
                ax.plot(rateE[:, amplitude_choice]*1e3, label='result')
                ax.plot(rateI[:, amplitude_choice]*1e3, label='result')
                ax.plot(I_input*1e3, label='input', alpha=0.5)
                ax = plt.subplot(323)
                ax.plot((rateE[:, amplitude_choice]-np.min(rateE[:, amplitude_choice]))/(np.max(rateE[:, amplitude_choice])-np.min(rateE[:, amplitude_choice]))*2-1, label='result')
                ax.plot(hist_filter/(np.max(hist_filter)-np.min(hist_filter))*2, '--', label='result filter')
                ax.plot(I_signal, label='Input', alpha=0.5)
                ax.legend()
                ax = plt.subplot(324)
                ax.plot(theta, label='result')
                ax.plot(theta_filter, '--', alpha=0.5, label='result filter')
                ax.plot(I_theta, alpha=0.5, label='Input')
                ax.legend()
                ax = plt.subplot(325)
                ax.plot(theta - I_theta, '.', markersize=0.5, label='result')
                ax.plot(theta_filter - I_theta, '.', markersize=0.5, label='result filter')
                ax.legend()


    # # Plot the excitatory and inhibitory signals, excitatory neuron adaptation, and noise input
    # # to the stimulated node
    # for i in range(50):
    #     plt.figure(figsize=(20,4))
    #     plt.rcParams.update({'font.size': 14})
    #     ax0 = plt.subplot(211)
    #     ax0.plot(result[0][0]*1e-3,result[0][1][:,0,i]*1e3, 'c')
    #     ax0.plot(result[0][0]*1e-3,result[0][1][:,1,i]*1e3, 'r')
    #     ax0.set_xlabel('time [s]')
    #     ax0.set_ylabel('firing rate [Hz]')
    #     ax1 = plt.subplot(212)
    #     ax1.plot(result[0][0]*1e-3,result[0][1][:,5,i], 'k')
    #     ax1.set_xlabel('time [s]')
    #     ax1.set_ylabel('adaptation [nA]')
    plt.show()