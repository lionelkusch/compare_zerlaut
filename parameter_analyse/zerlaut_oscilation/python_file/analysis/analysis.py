import numpy as np
import datetime
from scipy.signal import butter, lfilter, hilbert
from scipy import signal
from elephant.spectral import welch_psd
import parameter_analyse.zerlaut_oscilation.python_file.run.tools_simulation as tools
import json


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


def frequency_analysis(recorded_signal, fs):
    """
    return the dominant frequency
    :param recorded_signal: recorded signal
    :param fs: frequency of sampling
    :return:
    """
    welch_model = welch_psd(recorded_signal, frequency_resolution=1.0, fs=fs, )
    frequency_dominant = welch_model[0][np.argmax(welch_model[1])]
    return frequency_dominant


def PVL_analysis(recorded_signal, I_signal, frequency, fs, remove_start_hilbert=50000, remove_end=50000):
    # filtering
    b, a = signal.butter(1, [frequency - 0.1, frequency + 0.1], fs=fs, btype='band')
    hist_filter = signal.lfilter(b, a, recorded_signal)
    theta_filter = np.angle(signal.hilbert(hist_filter))

    PLV_value, PLV_angle = PLV(np.angle(signal.hilbert(I_signal))[remove_start_hilbert:-remove_end], theta_filter[remove_start_hilbert:-remove_end])
    return PLV_value, PLV_angle


def analysis(path_simulation, begin=0.0, end=20000.0, fs=1e4, sampling_ms=0.1,
             init_remove=50000):
    # take the data
    with open(path_simulation+'/parameter.json') as f:
            parameters = json.load(f)
    frequency = parameters['parameter_stimulation']["frequency"]
    rate = parameters['parameter_model']['initial_condition']["external_input_excitatory_to_excitatory"][0]
    noise = parameters['parameter_integrator']["noise_parameter"]["nsig"][0]
    result = tools.get_result(path_simulation, begin, end)
    times = result[0][0]
    rateE = result[0][1][:, 0, :]
    stdE = result[0][1][:, 2, :]
    rateI = result[0][1][:, 1, :]
    stdI = result[0][1][:, 4, :]
    corrEI = result[0][1][:, 3, :]
    adaptationE = result[0][1][:, 5, :]

    results = []
    for index, amplitude in enumerate(parameters['parameter_stimulation']["amp"]):
        # Input signal
        I_signal = np.sin(2 * np.pi * frequency * 1e3 * np.arange(begin, end, sampling_ms) * 1e-3)

        ex_frequency_dominant = frequency_analysis(rateE[:, index] * 1e3, fs=fs)
        in_frequency_dominant = frequency_analysis(rateI[:, index] * 1e3, fs=fs)

        if ex_frequency_dominant != frequency* 1e3:
            print('Bad max frequency ' + str(ex_frequency_dominant) +
                  ' => frequency : ' + str(frequency* 1e3) + ' noise : ' + str(noise) +
                  ' amplitude: ' + str(amplitude* 1e3))
            # print(welch_model[0][np.argmax(welch_model[1])])
            # raise Exception('bad frequency')

        if frequency > 0.0:
            ex_PLV_value, ex_PLV_angle = PVL_analysis(rateE[:, index] * 1e3, I_signal, frequency=frequency*1e3, fs=fs)
            in_PLV_value, in_PLV_angle = PVL_analysis(rateI[:, index] * 1e3, I_signal, frequency=frequency*1e3, fs=fs)
        else:
            ex_PLV_value, ex_PLV_angle = None, None
            in_PLV_value, in_PLV_angle = None, None

        result_ex = {
            'date':  datetime.datetime.now(),
            'path_file': path_simulation,
            'names_population': 'excitatory',
            'frequency_dom': ex_frequency_dominant,
            'PLV_value': ex_PLV_value,
            'PLV_angle': ex_PLV_angle,
            'max_rates': np.max(rateE[init_remove:, index] * 1e3),
            'min_rates': np.min(rateE[init_remove:, index] * 1e3),
            'mean_rates': np.mean(rateE[init_remove:, index] * 1e3),
            'std_rates': np.std(rateE[init_remove:, index] * 1e3),
            'amplitude': amplitude*1e3,
            'frequency': frequency*1e3,
            'rate': rate*1e3,
            'noise': noise
        }
        results.append(result_ex)
        result_in = {
            'date':  datetime.datetime.now(),
            'path_file': path_simulation,
            'names_population': 'inhibitory',
            'frequency_dom': in_frequency_dominant,
            'PLV_value': in_PLV_value,
            'PLV_angle': in_PLV_angle,
            'max_rates': np.max(rateI[init_remove:, index] * 1e3),
            'min_rates': np.min(rateI[init_remove:, index] * 1e3),
            'mean_rates': np.mean(rateI[init_remove:, index] * 1e3),
            'std_rates': np.std(rateI[init_remove:, index] * 1e3),
            'amplitude': amplitude*1e3,
            'frequency': frequency*1e3,
            'rate': rate*1e3,
            'noise': noise
        }
        results.append(result_in)
    return results