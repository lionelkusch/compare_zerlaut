import matplotlib.pyplot as plt
import os
import numpy as np
from parameter_analyse.static.python_file.plot.helper_function import get_gids_all, load_spike_all, slidding_window


def plot_histogram(hist_ex, hist_slide_ex, hist_in, hist_slide_in):
    """
    ompare histogram and smoothing version of it
    :param hist_ex: histogram of excitatory population
    :param hist_slide_ex: smooth version of histogram of excitatory population
    :param hist_in: histogram of inhibitory population
    :param hist_slide_in: smooth version of histogram of inhibitory population
    :return:
    """
    plt.figure()
    plt.plot(hist_ex[1][:-1], hist_ex[0])
    plt.plot(hist_in[1][:-1], hist_in[0])
    plt.figure()
    plt.plot(hist_slide_ex)
    plt.plot(hist_slide_in)
    plt.figure()
    plt.hist(hist_slide_ex)
    plt.figure()
    plt.hist(hist_slide_in)
    plt.show()


def get_mean_variance(path, begin, end, dt=0.1, window=5.0, plot_compare_hist=False):
    """
    generate mean, covariance and std of the histogram for excitatory and inhibitory population
    :param path: path of the folder
    :param begin: start analyse
    :param end: end analysis
    :param dt: integration step
    :param window: size of window for smoothing values
    :param plot_compare_hist: boolean for evaluate histogram and smoothing version
    :return: mean, std of excitatory and inhibitory and the covariance
    """
    gids_all = get_gids_all(path)
    nb_ex = gids_all['excitatory'][0][1] - gids_all['excitatory'][0][0]
    nb_in = gids_all['inhibitory'][0][1] - gids_all['inhibitory'][0][0]
    data_pop_all = load_spike_all(gids_all, path, begin, end)
    if data_pop_all == -1:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    hist_ex = np.histogram(data_pop_all['excitatory'][1], bins=int((end - begin) / dt))
    hist_slide_ex = slidding_window(hist_ex[0], int(window / dt)) / nb_ex / (dt * 1e-3)
    hist_in = np.histogram(data_pop_all['inhibitory'][1], bins=int((end - begin) / dt))
    hist_slide_in = slidding_window(hist_in[0], int(window / dt)) / nb_in / (dt * 1e-3)
    covariance = np.cov(np.stack((hist_slide_ex, hist_slide_in)))
    if plot_compare_hist:
        plot_histogram(hist_ex, hist_slide_ex, hist_in, hist_slide_in)
    return hist_slide_ex.mean(), covariance[0, 0], hist_slide_in.mean(), covariance[1, 1], covariance[0, 1]


def plot_mean_variance(path_init, rate_range, ax, b=60.0, font_size=10.0, tickfont_size=7.0,
                       begin=1000.0, end=5000.0, plot_covariance=False):
    """
    plot mean and variance dependant of the input firing rate
    :param path_init: path of the folder
    :param rate_range: range of external firing rate
    :param ax: axis for plotting the values
    :param b: value of b
    :param font_size: font size of labels
    :param tickfont_size: font size of the ticks
    :param begin: start the analysis
    :param end: end the analysis
    :param plot_covariance: plot the covariance if it's need it
    :return:
    """
    # generate a file with all the information because it take time to compute all of it
    if not os.path.exists(path_init + '/' + str(b) + '_mean_var.npy'):
        result = []
        for rate in rate_range:
            print(path_init + '/_b_' + str(b) + '_rate_' + str(float(rate)) + '/')
            result.append(
                get_mean_variance(path_init + '/_b_' + str(b) + '_rate_' + str(float(rate)) + '/', begin, end))
        np.save(path_init + '/' + str(b) + '_mean_var.npy', result)
    else:
        result = np.load(path_init + '/' + str(b) + '_mean_var.npy')
    result = np.concatenate([np.expand_dims(rate_range, axis=1), result], axis=1)

    # plot mean with dash line of covariance
    ax.plot(result[:, 0], result[:, 1], 'g')
    ax.plot(result[:, 0], np.array([result[:, 1] + np.sqrt(result[:, 2]), result[:, 1] - np.sqrt(result[:, 2])]).swapaxes(1,0), '--', alpha=0.5, color='g')
    # ax.fill_between(result[:, 0], result[:, 1] + result[:, 2], result[:, 1] - result[:, 2], alpha=0.5, color='g')
    ax.plot(result[:, 0], result[:, 3], 'r')
    # ax.fill_between(result[:, 0], result[:, 3] + result[:, 4], result[:, 3] - result[:, 4], alpha=0.5, color='r')
    ax.plot(result[:, 0], np.array([result[:, 3] + np.sqrt(result[:, 4]), result[:, 3] - np.sqrt(result[:, 4])]).swapaxes(1,0), '--', alpha=0.5, color='r')

    if plot_covariance:
        plt.figure()
        plt.plot(result[:, 0], result[:, 2], 'g')
        plt.plot(result[:, 0], result[:, 4], 'r')
        plt.plot(result[:, 0], result[:, 5], 'm')


if __name__ == "__main__":
    rate_range = range(100)
    path_init = os.path.dirname(os.path.realpath(__file__)) + "/../../simulation/"
    path = path_init + '/master_seed_0/'
    plt.figure(figsize=(20, 10))
    ax = plt.gca()
    plot_mean_variance(path, rate_range, ax, b=0.0, font_size=30.0, tickfont_size=20.0)
    plt.show()
    plt.figure(figsize=(20, 10))
    ax = plt.gca()
    plot_mean_variance(path, rate_range, ax, b=30.0, font_size=30.0, tickfont_size=20.0)
    plt.show()
    plt.figure(figsize=(20, 10))
    ax = plt.gca()
    plot_mean_variance(path, rate_range, ax, b=60.0, font_size=30.0, tickfont_size=20.0)
    plt.show()
