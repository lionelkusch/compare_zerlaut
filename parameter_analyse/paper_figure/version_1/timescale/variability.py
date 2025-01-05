import numpy as np
from parameter_analyse.static.python_file.plot.helper_function import get_gids_all, load_spike_all, slidding_window
from parameter_analyse.static.python_file.plot.print_study_time_std import gen_log_space
import pathos.multiprocessing as mp
import dill
import os
import quantities as pq
from elephant.spike_train_correlation import spike_train_timescale
from elephant.conversion import BinnedSpikeTrain


def test_function(path, begin, end, nb_sample=5, indexes_default=np.array(np.array([10000, 20000]), dtype=int)):
    """
    generate value of the mean and variance
    :param path: path of the mean and the variance
    :param begin: start of the analysis
    :param end: end of the analysis
    :param dt: step of integration
    :param nb_test: number of element for the window of the measure
    :param nb_cpu: number of cpu for parallel
    :param nb_sample: number of sample to get
    :return: array of mean and covariance
    """
    # get the data
    gids_all = get_gids_all(path)
    data_pop_all = load_spike_all(gids_all, path, begin, end)

    for title, dt in [['0.1', 0.1], ['1', 1.]]:
        for index in [500, 1000, 1500, 2000]:
            hist_ex = np.histogram(data_pop_all['excitatory'][1], bins=int((end - begin) / dt))
            hist_in = np.histogram(data_pop_all['inhibitory'][1], bins=int((end - begin) / dt))
            lag_range = np.arange(0, int(100/dt)+1, 1)
            if indexes_default is None:
                indexes = list(range(hist_ex[0].shape[0] - index))
                np.random.shuffle(indexes)
            else:
                indexes = indexes_default / dt

            for reduce in indexes[:nb_sample]:
                # plot histogram
                hist_bin_hist_ex = BinnedSpikeTrain(np.expand_dims(hist_ex[0][reduce:reduce + int(index/dt)], 0), t_start=0 * pq.ms,
                                                    t_stop=index * pq.ms, bin_size=dt * pq.ms)
                import matplotlib.pyplot as plt
                plt.figure()
                plt.plot(np.arange(0, index)/10, hist_ex[0][reduce:reduce+index])
                # plot auocorrelation
                from elephant.spike_train_correlation import cross_correlation_histogram
                res = cross_correlation_histogram(hist_bin_hist_ex, hist_bin_hist_ex, binary=False, border_correction=True,
                                                  window=[-lag_range[-1], lag_range[-1]], cross_correlation_coefficient=True)
                plt.figure()
                plt.plot(np.expand_dims(np.array(res[1]), 1)[len(lag_range)-1:]*dt, np.array(res[0])[len(lag_range)-1:])
                for timescale_constant in np.arange(1, 20, 1):
                    timescale = spike_train_timescale(hist_bin_hist_ex, max_tau=timescale_constant * pq.ms)
                    plt.plot(lag_range*dt, np.exp(-lag_range*dt/float(timescale)), label=str(timescale_constant)+ ' '+ str(timescale))
                plt.legend()
                plt.title(title+' '+str(reduce*dt))
                # plot second estimation
                from scipy.signal import find_peaks
                signal = np.array(res[0])[len(lag_range)-1:, 0]
                time_lag = np.array(res[1])[len(lag_range) - 1:] * dt
                peaks, properties = find_peaks(signal, height=0)# threshold=0.001)

                from scipy.optimize import curve_fit
                def funct(x, b):
                    return np.exp(-b*x)

                popt, pcov = curve_fit(funct, time_lag[peaks], signal[peaks])

                plt.figure()
                plt.plot(time_lag, signal)
                plt.plot(time_lag[peaks], signal[peaks], '.')
                plt.plot(time_lag, funct(time_lag, popt[0]) )
                plt.title(title+' '+str(index)+' '+str(reduce*dt)+' '+str(popt[0]*dt))
    plt.show()

from elephant.spike_train_correlation import spike_train_timescale, cross_correlation_histogram
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
def decay_exponential(x, b):
    "decay exponetial function with one parameters"
    return np.exp(-b * x)

def time_scale(hist_bin, dt=0.1, lag=100, duration=1000, binarize=False, bordercorrection=True,
               cross_correlation_coefficient=True):
    """
    compute the time scale based on an histogram based on fitting an exponential decay over the autocorrelation coefficients
    :param hist: histogram
    :param dt: unit of the bin of the histogram
    :param lag: the lag of the crosscorrelation
    :param duration: duration of the histogram
    :return:
    """
    lag_range = np.arange(0, int(lag / dt) + 1, 1)
    autocorrelation = cross_correlation_histogram(hist_bin, hist_bin, binary=binarize,
                                                  border_correction=bordercorrection,
                                                  window=[-lag_range[-1], lag_range[-1]],
                                                  cross_correlation_coefficient=cross_correlation_coefficient)
    coefficients = np.array(autocorrelation[0])[len(lag_range) - 1:, 0]
    time_lag = np.array(autocorrelation[1])[len(lag_range) - 1:] * dt #error need to remove dt
    peaks, properties = find_peaks(coefficients, height=0)
    if len(peaks) != 0:
        try :
            popt, pcov = curve_fit(decay_exponential, time_lag[peaks], coefficients[peaks])
        except RuntimeError:
            popt = [-1.]
    else:
        popt = [-1.]

    return coefficients, time_lag, popt[0]

def get_time_scale_long(path, begin, end, dt=0.1, nb_test=50, nb_cpu=16, nb_sample=50000, lag=100):
    """
    generate value of the mean and variance
    :param path: path of the mean and the variance
    :param begin: start of the analysis
    :param end: end of the analysis
    :param dt: step of integration
    :param nb_test: number of element for the window of the measure
    :param nb_cpu: number of cpu for parallel
    :param nb_sample: number of sample to get
    :return: array of mean and covariance
    """
    # get the data
    gids_all = get_gids_all(path)
    data_pop_all = load_spike_all(gids_all, path, begin, end)
    if data_pop_all == -1:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    # smooth the histogram
    hist_ex_0_1 = np.histogram(data_pop_all['excitatory'][1], bins=int((end - begin)/dt))
    hist_in_0_1 = np.histogram(data_pop_all['inhibitory'][1], bins=int((end - begin)/dt))
    hist_ex_1 = np.histogram(data_pop_all['excitatory'][1], bins=int((end - begin)))
    hist_in_1 = np.histogram(data_pop_all['inhibitory'][1], bins=int((end - begin)))

    def analyse_function(arg): # lag=100
        """
        get the mean and the covariance for size of the measure
        :param arg:
        :return:
        """
        print(arg)
        index = arg
        timescale = [[], [], [], []]
        if hist_ex_1[0].shape[0] - index < nb_sample:
            #  take all the values
            for reduce in range(hist_ex_1[0].shape[0] - index):
                print(reduce, arg, len(timescale), len(timescale[0]), len(timescale[1]), len(timescale[2]), len(timescale[3]))
                hist_0_1_bin_hist_ex = BinnedSpikeTrain(np.expand_dims(hist_ex_0_1[0][reduce:reduce + index*10], 0), t_start=0 * pq.ms,
                                                     t_stop=index * pq.ms, bin_size=0.1 * pq.ms)
                hist_0_1_bin_hist_in = BinnedSpikeTrain(np.expand_dims(hist_in_0_1[0][reduce:reduce + index*10], 0), t_start=0 * pq.ms,
                                                     t_stop=index * pq.ms, bin_size=0.1 * pq.ms)
                timescale[0].append(time_scale(hist_0_1_bin_hist_ex, duration=index, dt=0.1, lag=lag)[2])
                timescale[1].append(time_scale(hist_0_1_bin_hist_in, duration=index, dt=0.1, lag=lag)[2])
                hist_1_bin_hist_ex = BinnedSpikeTrain(np.expand_dims(hist_ex_1[0][reduce:reduce + index], 0), t_start=0 * pq.ms,
                                                        t_stop=index * pq.ms, bin_size=1 * pq.ms)
                hist_1_bin_hist_in = BinnedSpikeTrain(np.expand_dims(hist_in_1[0][reduce:reduce + index], 0), t_start=0 * pq.ms,
                                                        t_stop=index * pq.ms, bin_size=1 * pq.ms)
                timescale[2].append(time_scale(hist_1_bin_hist_ex, duration=index, dt=1., lag=lag)[2])
                timescale[3].append(time_scale(hist_1_bin_hist_in, duration=index, dt=1., lag=lag)[2])
            print(arg, 'end', timescale[0][-1], timescale[1][-1], timescale[2][-1], timescale[3][-1])
        else:
            # choice a reduce number of sample
            indexes = list(range(hist_ex_1[0].shape[0] - index))
            np.random.shuffle(indexes)
            for reduce in indexes[:nb_sample]:
                print(reduce, arg, len(timescale), len(timescale[0]), len(timescale[1]), len(timescale[2]), len(timescale[3]))
                # if 0 > index/10 or np.expand_dims(hist_ex_0_1[0][reduce:reduce + index], 0).max() == 0:
                #     print(index)
                hist_0_1_bin_hist_ex = BinnedSpikeTrain(np.expand_dims(hist_ex_0_1[0][reduce:reduce + index*10], 0), t_start=0 * pq.ms,
                                                        t_stop=index * pq.ms, bin_size=0.1 * pq.ms)
                hist_0_1_bin_hist_in = BinnedSpikeTrain(np.expand_dims(hist_in_0_1[0][reduce:reduce + index*10], 0), t_start=0 * pq.ms,
                                                        t_stop=index * pq.ms, bin_size=0.1 * pq.ms)
                timescale[0].append(time_scale(hist_0_1_bin_hist_ex, duration=index, dt=0.1)[2])
                timescale[1].append(time_scale(hist_0_1_bin_hist_in, duration=index, dt=0.1)[2])
                hist_1_bin_hist_ex = BinnedSpikeTrain(np.expand_dims(hist_ex_1[0][reduce:reduce + int(index)], 0), t_start=0 * pq.ms,
                                                      t_stop=index * pq.ms, bin_size=1 * pq.ms)
                hist_1_bin_hist_in = BinnedSpikeTrain(np.expand_dims(hist_in_1[0][reduce:reduce + int(index)], 0), t_start=0 * pq.ms,
                                                      t_stop=index * pq.ms, bin_size=1 * pq.ms)
                timescale[2].append(time_scale(hist_1_bin_hist_ex, duration=index, dt=1)[2])
                timescale[3].append(time_scale(hist_1_bin_hist_in, duration=index, dt=1)[2])
            print(arg, 'reduce sample end', timescale[0][-1], timescale[1][-1], timescale[2][-1], timescale[3][-1])
        return timescale

    p = mp.ProcessingPool(ncpus=nb_cpu)
    res = p.map(dill.copy(analyse_function), gen_log_space(hist_ex_1[0].shape[0] - nb_sample * dt - lag*2, nb_test)+int(lag*2))
    # concatenate all the result
    timescale_ex_0_1 = []
    timescale_in_0_1 = []
    timescale_ex_1 = []
    timescale_in_1 = []
    for i in range(len(res)):
        print(hist_ex_1[0].shape[0] - len(res[i][0]))
        timescale_ex_0_1.append(res[i][0])
        timescale_in_0_1.append(res[i][1])
        timescale_ex_1.append(res[i][2])
        timescale_in_1.append(res[i][3])

    return [timescale_ex_0_1, timescale_in_0_1, timescale_ex_1, timescale_in_1]


# def get_time_scale_long(path, begin, end, dt=0.1, nb_test=50, nb_cpu=16, nb_sample=50000):
#     """
#     generate value of the mean and variance
#     :param path: path of the mean and the variance
#     :param begin: start of the analysis
#     :param end: end of the analysis
#     :param dt: step of integration
#     :param nb_test: number of element for the window of the measure
#     :param nb_cpu: number of cpu for parallel
#     :param nb_sample: number of sample to get
#     :return: array of mean and covariance
#     """
#     # get the data
#     gids_all = get_gids_all(path)
#     data_pop_all = load_spike_all(gids_all, path, begin, end)
#     if data_pop_all == -1:
#         return 0.0, 0.0, 0.0, 0.0, 0.0
#     # smooth the histogram
#     hist_ex = np.histogram(data_pop_all['excitatory'][1], bins=int((end - begin)/dt))
#     hist_in = np.histogram(data_pop_all['inhibitory'][1], bins=int((end - begin)/dt))
#
#     def analyse_function(arg, lag=10):
#         """
#         get the mean and the covariance for size of the measure
#         :param arg:
#         :return:
#         """
#         print(arg)
#         index = arg
#         timescale = [[], []]
#         if hist_ex[0].shape[0] - index < nb_sample:
#             #  take all the values
#             for reduce in range(hist_ex[0].shape[0] - index):
#                 hist_0_1_bin_hist_ex = BinnedSpikeTrain(np.expand_dims(hist_ex[0][reduce:reduce + index], 0), t_start=0 * pq.ms,
#                                                      t_stop=index/10 * pq.ms, bin_size=0.1 * pq.ms)
#                 hist_0_1_bin_hist_in = BinnedSpikeTrain(np.expand_dims(hist_in[0][reduce:reduce + index], 0), t_start=0 * pq.ms,
#                                                      t_stop=index/10 * pq.ms, bin_size=0.1 * pq.ms)
#                 timescale[0].append(spike_train_timescale(hist_0_1_bin_hist_ex, max_tau=lag * pq.ms))
#                 timescale[1].append(spike_train_timescale(hist_0_1_bin_hist_in, max_tau=lag * pq.ms))
#             print(arg, 'end', timescale[0], timescale[1])
#         else:
#             # choice a reduce number of sample
#             indexes = list(range(hist_ex[0].shape[0] - index))
#             np.random.shuffle(indexes)
#             for reduce in indexes[:nb_sample]:
#                 if 0 > index/10 or np.expand_dims(hist_ex[0][reduce:reduce + index], 0).max() == 0:
#                     print(index)
#                 hist_0_1_bin_hist_ex = BinnedSpikeTrain(np.expand_dims(hist_ex[0][reduce:reduce + index], 0), t_start=0 * pq.ms,
#                                                         t_stop=index/10 * pq.ms, bin_size=0.1 * pq.ms)
#                 hist_0_1_bin_hist_in = BinnedSpikeTrain(np.expand_dims(hist_in[0][reduce:reduce + index], 0), t_start=0 * pq.ms,
#                                                         t_stop=index/10 * pq.ms, bin_size=0.1 * pq.ms)
#                 timescale[0].append(spike_train_timescale(hist_0_1_bin_hist_ex, max_tau=lag * pq.ms))
#                 timescale[1].append(spike_train_timescale(hist_0_1_bin_hist_in, max_tau=lag * pq.ms))
#             print(arg, 'reduce sample end',
#                   np.min(timescale[0]), np.max(timescale[0]), np.mean(timescale[0]),
#                   np.min(timescale[1]), np.max(timescale[1]), np.mean(timescale[1]))
#         return timescale
#
#     p = mp.ProcessingPool(ncpus=nb_cpu)
#     res = p.map(dill.copy(analyse_function), gen_log_space(hist_ex[0].shape[0] - nb_sample * dt, nb_test))
#     # concatenate all the result
#     timescale_ex = []
#     timescale_in = []
#     for i in range(len(res)):
#         print(hist_ex[0].shape[0] - len(res[i][0]))
#         timescale_ex.append(res[i][0])
#         timescale_in.append(res[i][1])
#
#     return [timescale_ex, timescale_in]

def generate_timescale_long(path_init, b=60.0, begin=1000.0, end=5000.0, rate_range=range(1, 100),
                               dt=0.1, window=5.0, nb_test=50, nb_sample=50000):
    """
    save mean, variance and covariance for an exploration of external input
    :param path_init: path of the folders
    :param b: values of b
    :param rate_range: range of firing rate
    :param begin: start of the analysis
    :param end: end of the analysis
    :param dt: step of integration
    :param window: size of the windows for smoothing the histogram
    :param nb_test: number of element for the window of the measure
    :param nb_sample: number of sample to get
    :return:
    """
    if not os.path.exists(path_init + '/' + str(b) + '_variance_timescale_1.npy'):
        for rate in rate_range:
            print(path_init + '/_b_' + str(b) + '_rate_' + str(float(rate)) + '/')
            result = get_time_scale_long(path_init + '/_b_' + str(b) + '_rate_' + str(float(rate)) + '/', begin, end,
                                       dt=dt, nb_test=nb_test, nb_sample=nb_sample)
            np.save(path_init + '/' + str(b) + '_variance_timescale_' + str(rate) + '_1.npi', result)

if __name__ == '__main__':
    path_init = os.path.dirname(os.path.realpath(__file__)) + "/../../../static/simulation/data/"
    path = path_init + '/long/'
    generate_timescale_long(path, b=0.0, rate_range=[10.0, 50.0, 60.0], begin=10000.0, end=40000.0,
                                    nb_test=50, nb_sample=50000)
    generate_timescale_long(path, b=30.0, rate_range=[10.0, 50.0, 60.0], begin=10000.0, end=40000.0,
                                    nb_test=50, nb_sample=50000)
    generate_timescale_long(path, b=60.0, rate_range=[10.0, 50.0, 60.0], begin=10000.0, end=40000.0,
                                    nb_test=50, nb_sample=50000)
