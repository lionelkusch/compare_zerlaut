#  Copyright 2023 Aix-Marseille Université
# "Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements; and to You under the Apache License, Version 2.0. "
import matplotlib.pyplot as plt
import os
import pathos.multiprocessing as mp
import dill
import numpy as np
from parameter_analyse.static.python_file.plot.helper_function import get_gids_all, load_spike_all, slidding_window


def gen_log_space(limit, n):
    """
    generate log space with integer
    code from
    https://stackoverflow.com/questions/12418234/logarithmically-spaced-integers
    :param limit: limit of values
    :param n: number of element
    :return:
    """
    result = [1]
    if n > 1:  # just a check to avoid ZeroDivisionError
        ratio = (float(limit) / result[-1]) ** (1.0 / (n - len(result)))
    while len(result) < n:
        next_value = result[-1] * ratio
        if next_value - result[-1] >= 1:
            # safe zone. next_value will be a different integer
            result.append(next_value)
        else:
            # problem! same integer. we need to find next_value by artificially incrementing previous value
            result.append(result[-1] + 1)
            # recalculate the ratio so that the remaining values will scale correctly
            ratio = (float(limit) / result[-1]) ** (1.0 / (n - len(result)))
    # round, re-adjust to 0 indexing (i.e. minus 1) and return np.uint64 array
    return np.array(list(map(lambda x: round(x) - 1, result)), dtype=int)


def get_mean_variance(path, begin, end, dt=0.1, window=5.0, nb_test=50, nb_cpu=16, nb_sample=50000, print_result=False):
    """
    generate value of the mean and variance
    :param path: path of the mean and the variance
    :param begin: start of the analysis
    :param end: end of the analysis
    :param dt: step of integration
    :param window: size of the windows for smoothing the histogram
    :param nb_test: number of element for the window of the measure
    :param nb_cpu: number of cpu for parallel
    :param nb_sample: number of sample to get
    :param print_result: print the result
    :return: array of mean and covariance
    """
    # get the data
    gids_all = get_gids_all(path)
    nb_ex = gids_all['excitatory'][0][1] - gids_all['excitatory'][0][0]
    nb_in = gids_all['inhibitory'][0][1] - gids_all['inhibitory'][0][0]
    data_pop_all = load_spike_all(gids_all, path, begin, end)
    if data_pop_all == -1:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    # smooth the histogram
    hist_ex = np.histogram(data_pop_all['excitatory'][1], bins=int((end - begin) / dt))
    hist_slide_ex = slidding_window(hist_ex[0], int(window / dt)) / nb_ex / (dt * 1e-3)
    hist_in = np.histogram(data_pop_all['inhibitory'][1], bins=int((end - begin) / dt))
    hist_slide_in = slidding_window(hist_in[0], int(window / dt)) / nb_in / (dt * 1e-3)

    def analyse_function(arg):
        """
        get the mean and the covariance for size of the measure
        :param arg:
        :return:
        """
        print(arg)
        index = arg
        covariance = []
        mean = []
        for reduce in range(hist_slide_ex.shape[0] - index):
            covariance.append(np.cov(np.stack((hist_slide_ex[reduce:reduce + index],
                                               hist_slide_in[reduce:reduce + index]))))
            mean.append([hist_slide_ex[reduce:reduce + index].mean(),
                         hist_slide_in[reduce:reduce + index].mean()])
        print(arg, 'end')
        return mean, covariance

    p = mp.ProcessingPool(ncpus=nb_cpu)
    res = p.map(dill.copy(analyse_function), gen_log_space(hist_slide_ex.shape[0] - int(window / dt) - nb_sample * dt,
                                                           nb_test))
    # concatenate all the result
    covariance = []
    mean = []
    for i in range(len(res)):
        print(hist_slide_ex.shape[0] - len(res[i][0]))
        mean.append(res[i][0])
        covariance.append(res[i][1])

    if print_result:
        print(res,
              (np.concatenate([mean[-1]])[:, 0].min(),
               np.concatenate([mean[-1]])[:, 0].max()),
              (np.concatenate([mean[-1]])[:, 1].min(),
               np.concatenate([mean[-1]])[:, 1].max()),
              (np.concatenate([covariance[-1]])[:, 0, 0].min(),
               np.concatenate([covariance[-1]])[:, 0, 0].max()),
              (np.concatenate([covariance[-1]])[:, 1, 0].min(),
               np.concatenate([covariance[-1]])[:, 1, 0].max()),
              (np.concatenate([covariance[-1]])[:, 1, 1].min(),
               np.concatenate([covariance[-1]])[:, 1, 1].max()),
              )
    return mean, covariance


def generate_variance_mean_std(path_init, b=60.0, begin=1000.0, end=5000.0, rate_range=range(1, 100),
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
    if not os.path.exists(path_init + '/' + str(b) + '_size_variance.npy'):
        for rate in rate_range:
            print(path_init + '/_b_' + str(b) + '_rate_' + str(float(rate)) + '/')
            result = get_mean_variance(path_init + '/_b_' + str(b) + '_rate_' + str(float(rate)) + '/', begin, end,
                                       dt=dt, window=window, nb_test=nb_test, nb_sample=nb_sample)
            np.save(path_init + '/' + str(b) + '_size_variance_rate_' + str(rate) + '.npi', result)


def plot_variance_mean_std_rate(path_init, range_rate=range(1, 100), b=60.0, begin=1000.0, end=5000.0,
                                dt=0.1, window=5.0, nb_test=50, nb_sample=5000):
    """
    plot mean and std distribution in time
    :param path_init: path of folder
    :param range_rate: rate of range to analyse
    :param b: value of b
    :param begin: start analysis
    :param end: end analysis
    :param dt: step integration
    :param window: window of smoothing
    :param nb_test: number of test
    :param nb_sample: number of sample to get
    :return:
    """
    # values of range of data
    values = gen_log_space(int((end - begin) / dt - window / dt) - int(window / dt) - nb_sample * dt, nb_test)
    for rate in range_rate:
        # get data
        print(rate)
        result = np.load(path_init + '/' + str(b) + '_size_variance_rate_' + str(rate) + '.npy', allow_pickle=True)
        means = np.concatenate([result[0, :]])
        mean_ex = []
        mean_in = []
        for mean in means:
            mean_ex.append(np.array(mean, dtype=float)[:, 0])
            mean_in.append(np.array(mean, dtype=float)[:, 1])
        covariances = np.concatenate([result[1, :]])
        variances_ex = []
        variances_in = []
        covariance = []
        for cov in covariances:
            variances_ex.append(np.array(cov, dtype=float)[:, 0, 0])
            variances_in.append(np.array(cov, dtype=float)[:, 1, 1])
            covariance.append(np.array(cov, dtype=float)[:, 1, 0])
        CV_ex = []
        CV_in = []
        for mean, cov in zip(means, covariances):
            CV_ex.append(np.array(cov, dtype=float)[:, 0, 0] / np.array(mean, dtype=float)[:, 0])
            CV_in.append(np.array(cov, dtype=float)[:, 1, 1] / np.array(mean, dtype=float)[:, 1])

        # plot data in one figure
        fig, axs = plt.subplots(1, 4, figsize=(20, 20))
        plot_violin(axs[0], mean_ex, values)
        plot_violin(axs[0], mean_in, values)
        axs[0].set_title('mean firing rate')
        plot_violin(axs[1], variances_ex, values)
        axs[1].set_title('variance excitatory')
        plot_violin(axs[2], covariance, values)
        axs[2].set_title('covariance')
        plot_violin(axs[3], variances_in, values)
        axs[3].set_title('variance inhibitory')
        # plt.subplots_adjust(top=0.97, bottom=0.08, left=0.03, right=0.91, hspace=0.2, wspace=0.1)
        # plt.savefig(path_init + '/' + str(b) + '_size_variance_rate_' + str(rate) + '.png')

        # coefficient of variation data
        fig, axs = plt.subplots(1, 2, figsize=(20, 20))
        plot_violin(axs[0], CV_ex, values)
        axs[0].set_title('coefficent fo variation excitatory')
        plot_violin(axs[1], CV_in, values)
        axs[1].set_title('coefficient of variation')

        # plot individual values
        fig, axs = plt.subplots(1, 1, figsize=(20, 20))
        plot_violin(axs, mean_ex, values)
        fig, axs = plt.subplots(1, 1, figsize=(20, 20))
        plot_violin(axs, mean_in, values)
        fig, axs = plt.subplots(1, 1, figsize=(20, 20))
        plot_violin(axs, variances_ex, values)
        fig, axs = plt.subplots(1, 1, figsize=(20, 20))
        plot_violin(axs, variances_in, values)

        plt.show()
        plt.close('all')


def get_mean_variance_long(path, begin, end, dt=0.1, window=5.0, nb_test=50, nb_sample=10000, nb_cpu=16):
    """
    generate value of the mean and variance for longer simulation
    :param path: path of the mean and the variance
    :param begin: start of the analysis
    :param end: end of the analysis
    :param dt: step of integration
    :param window: size of the windows for smoothing the histogram
    :param nb_test: number of element for the window of the measure
    :param nb_sample: number of maximum of sample by measure
    :param nb_cpu: number of cpu for parallel
    :return: array of mean and covariance
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

    def analyse_function(arg):
        """
        get the mean and the covariance for size of the measure
        :param arg:
        :return:
        """
        print(arg)
        index = arg
        covariance = []
        mean = []
        if hist_slide_ex.shape[0] - index < nb_sample:
            #  take all the values
            for reduce in range(hist_slide_ex.shape[0] - index):
                covariance.append(
                    np.cov(np.stack((hist_slide_ex[reduce:reduce + index], hist_slide_in[reduce:reduce + index]))))
                mean.append([hist_slide_ex[reduce:reduce + index].mean(), hist_slide_in[reduce:reduce + index].mean()])
            print(arg, 'end')
        else:
            # choice a reduce number of sample
            indexes = list(range(hist_slide_ex.shape[0] - index))
            np.random.shuffle(indexes)
            for reduce in indexes[:nb_sample]:
                covariance.append(
                    np.cov(np.stack((hist_slide_ex[reduce:reduce + index], hist_slide_in[reduce:reduce + index]))))
                mean.append([hist_slide_ex[reduce:reduce + index].mean(), hist_slide_in[reduce:reduce + index].mean()])
            print(arg, 'reduce sample end')
        return mean, covariance

    p = mp.ProcessingPool(ncpus=nb_cpu)
    values = gen_log_space(hist_slide_ex.shape[0] - int(window / dt) - nb_sample * dt, nb_test)
    res = p.map(dill.copy(analyse_function), values)
    # concatenate the result
    covariance = []
    mean = []
    for i in range(len(res)):
        print(hist_slide_ex.shape[0] - len(res[i][0]))
        mean.append(res[i][0])
        covariance.append(res[i][1])
    return mean, covariance


def generate_variance_mean_std_long(path_init, b=60.0, begin=1000.0, end=5000.0,
                                    rate_range=range(1, 100), dt=0.1, window=5.0, nb_test=50,
                                    nb_sample=10000):
    """
    save mean, variance and covariance for an exploration of external input for long simulation
    :param path_init: path of the folders
    :param b: values of b
    :param rate_range: range of firing rate
    :param begin: start of the analysis
    :param end: end of the analysis
    :param dt: step of integration
    :param window: size of the windows for smoothing the histogram
    :param nb_test: number of element for the window of the measure
    :param nb_sample: number of maximal of sample by measure
    :return:
    """
    if not os.path.exists(path_init + '/' + str(b) + '_size_variance.npy'):
        for rate in rate_range:
            print(path_init + '/_b_' + str(b) + '_rate_' + str(float(rate)) + '/')
            result = get_mean_variance_long(path_init + '/_b_' + str(b) + '_rate_' + str(float(rate)) + '/', begin, end,
                                            dt=dt, window=window, nb_test=nb_test, nb_sample=nb_sample)
            np.save(path_init + '/' + str(b) + '_size_variance_rate_' + str(rate) + '.npy', result)


def plot_violin(ax, data, range_values):
    """
    violin plot of the result
    :param ax: axis of the figure
    :param data: data of plotting
    :param range_values: range of values of the measure window
    :return:
    """
    violin_parts = ax.violinplot(data, widths=0.5, showmeans=True, showmedians=True)
    for range_value, mean in zip(range(len(range_values)), data):
        ax.scatter(np.repeat(range_value + .8, len(mean)), mean, color='black', s=0.1)
        ax.scatter(range_value + 1, mean.mean(), marker='o', color='r')
    for pc in violin_parts['bodies']:
        pc.set_facecolor('b')
        pc.set_edgecolor('black')
    ax.hlines(data[-1][0], xmin=1, xmax=len(range_values), color='r', alpha=0.5)
    ax.set_xticks(np.arange(1, len(range_values) + 1))
    ax.set_xticklabels(range_values)
    ax.tick_params(axis='x', labelrotation=90)


def plot_variance_mean_std_unique_long(path_init, b=60.0, rate=10, begin=1000.0, end=5000.0,
                                       dt=0.1, window=5.0, nb_test=50, nb_sample=50000, plt_save_data=False):
    """
    plot violin mean and std long
    :param path_init: path of folder
    :param b: b
    :param rate: rate
    :param begin: begin analysis
    :param end: end analysis
    :param dt: step integration
    :param window: window for smoothing
    :param nb_test: number of test
    :param nb_sample: number of sample
    :param plt_save_data: boolean for saving or not the figure
    :return:
    """
    values = gen_log_space(int((end - begin) / dt - window / dt) - int(window / dt) - nb_sample * dt, nb_test)
    print(rate)
    # get data
    result = np.load(path_init + '/' + str(b) + '_size_variance_rate_' + str(rate) + '.npy', allow_pickle=True)
    means = np.concatenate([result[0, :]])
    mean_ex = []
    mean_in = []
    for mean in means:
        mean_ex.append(np.array(mean, dtype=float)[:, 0])
        mean_in.append(np.array(mean, dtype=float)[:, 1])
    covariances = np.concatenate([result[1, :]])
    variances_ex = []
    variances_in = []
    covariance = []
    for cov in covariances:
        variances_ex.append(np.array(cov, dtype=float)[:, 0, 0])
        variances_in.append(np.array(cov, dtype=float)[:, 1, 1])
        covariance.append(np.array(cov, dtype=float)[:, 1, 0])
    CV_ex = []
    CV_in = []
    for mean, cov in zip(means, covariances):
        CV_ex.append(np.array(cov, dtype=float)[:, 0, 0] / np.array(mean, dtype=float)[:, 0])
        CV_in.append(np.array(cov, dtype=float)[:, 1, 1] / np.array(mean, dtype=float)[:, 1])

    # plot all values togethers
    # fig, axs = plt.subplots(1, 4, figsize=(20, 20))
    # plot_violin(axs[0], mean_ex, values)
    # plot_violin(axs[0], mean_in, values)
    # axs[0].set_title('mean firing rate')
    # plot_violin(axs[1], variances_ex, values)
    # axs[1].set_title('variance excitatory')
    # plot_violin(axs[2], covariance, values)
    # axs[2].set_title('covariance')
    # plot_violin(axs[3], variances_in, values)
    # axs[3].set_title('variance inhibitory')
    # plt.subplots_adjust(top=0.97, bottom=0.08, left=0.03, right=0.91, hspace=0.2, wspace=0.1)

    # fig, axs = plt.subplots(1, 2, figsize=(20, 20))
    # plot_violin(axs[0], CV_ex, values)
    # axs[0].set_title('coefficent fo variation excitatory')
    # plot_violin(axs[1], CV_in, values)
    # axs[1].set_title('coefficient of variation')

    # plot values in figures
    fig, axs = plt.subplots(1, 1, figsize=(20, 20))
    plot_violin(axs, mean_ex, values)
    plt.savefig(path_init + '/' + str(b) + '_size_variance_rate_' + str(rate) + '_mean_ex.png')
    fig, axs = plt.subplots(1, 1, figsize=(20, 20))
    plot_violin(axs, mean_in, values)
    plt.savefig(path_init + '/' + str(b) + '_size_variance_rate_' + str(rate) + '_mean_in.png')
    fig, axs = plt.subplots(1, 1, figsize=(20, 20))
    plot_violin(axs, variances_ex, values)
    plt.savefig(path_init + '/' + str(b) + '_size_variance_rate_' + str(rate) + '_std_ex.png')
    fig, axs = plt.subplots(1, 1, figsize=(20, 20))
    plot_violin(axs, covariance, values)
    plt.savefig(path_init + '/' + str(b) + '_size_variance_rate_' + str(rate) + '_cov.png')
    fig, axs = plt.subplots(1, 1, figsize=(20, 20))
    plot_violin(axs, variances_in, values)

    if plt_save_data:
        plt.savefig(path_init + '/' + str(b) + '_size_variance_rate_' + str(rate) + '_std_in.png')
    else:
        plt.show()

    plt.close('all')


if __name__ == "__main__":
    path_init = os.path.dirname(os.path.realpath(__file__)) + "/../../simulation/data/"
    path = path_init + '/master_seed_0/'
    plt.figure(figsize=(20, 20))
    ax = plt.gca()
    # generate_variance_mean_std(path, ax, b=0.0, font_size=30.0, tickfont_size=20.0, burst=False, size_mark=1.0)
    # plot_variance_mean_std_unique(path, ax, b=0.0, font_size=30.0, tickfont_size=20.0, burst=False, size_mark=1.0)
    # plt.show()
    # plt.figure(figsize=(20, 10))
    # ax = plt.gca()
    # generate_variance_mean_std(path, ax, b=30.0, font_size=30.0, tickfont_size=20.0, burst=False, size_mark=1.0)
    # plot_variance_mean_std_unique(path, ax, b=30.0, font_size=30.0, tickfont_size=20.0, burst=False, size_mark=1.0)
    # # plt.show()
    # plt.figure(figsize=(20, 10))
    # ax = plt.gca()
    # generate_variance_mean_std(path, ax, b=60.0, font_size=30.0, tickfont_size=20.0, burst=False, size_mark=1.0)
    # plot_variance_mean_std_unique(path, ax, b=60.0, font_size=30.0, tickfont_size=20.0, burst=False, size_mark=1.0)
    # plt.show()
    # plt.subplots_adjust(top=0.99, bottom=0.11, left=0.05, right=0.98, hspace=0.2, wspace=0.2)
    # plt.savefig(path_figure+'/SP_figure_29.pdf', dpi=300)

    ## long simulation
    path_init = os.path.dirname(os.path.realpath(__file__)) + "/../../simulation/data/"
    path = path_init + '/long/'
    generate_variance_mean_std_long(path, b=0.0, rate_range=[10.0, 50.0, 60.0], begin=10000.0, end=40000.0,
                                    nb_test=50, nb_sample=50000)
    # plot_variance_mean_std_unique_long(path, b=0.0, rate=10.0, nb_test=200)
    # plot_variance_mean_std_unique_long(path, b=0.0, rate=50.0, font_size=30.0, tickfont_size=20.0, burst=False,
    #                                    size_mark=1.0, nb_test=200)
    # plot_variance_mean_std_unique_long(path, b=0.0, rate=60.0, font_size=30.0, tickfont_size=20.0, burst=False,
    #                                    size_mark=1.0, nb_test=200)
    generate_variance_mean_std_long(path, b=30.0, rate_range=[10.0, 50.0, 60.0], begin=10000.0, end=40000.0,
                                    nb_test=50, nb_sample=50000)
    # plot_variance_mean_std_unique_long(path, ax, b=30.0, rate=10.0, font_size=30.0, tickfont_size=20.0, burst=False,
    #                                    size_mark=1.0, nb_test=200)
    # plot_variance_mean_std_unique_long(path, ax, b=30.0, rate=50.0, font_size=30.0, tickfont_size=20.0, burst=False,
    #                                    size_mark=1.0, nb_test=200)
    # plot_variance_mean_std_unique_long(path, ax, b=30.0, rate=60.0, font_size=30.0, tickfont_size=20.0, burst=False,
    #                                    size_mark=1.0, nb_test=200)
    generate_variance_mean_std_long(path, b=60.0, rate_range=[10.0, 50.0, 60.0], begin=10000.0, end=40000.0,
                                    nb_test=50, nb_sample=50000)
    # plot_variance_mean_std_unique_long(path, ax, b=60.0, rate=10.0, font_size=30.0, tickfont_size=20.0, burst=False,
    #                                    size_mark=1.0, nb_test=200)
    # plot_variance_mean_std_unique_long(path, ax, b=60.0, rate=50.0, font_size=30.0, tickfont_size=20.0, burst=False,
    #                                    size_mark=1.0, nb_test=200)
    # plot_variance_mean_std_unique_long(path, ax, b=60.0, rate=60.0, font_size=30.0, tickfont_size=20.0, burst=False,
    #                                    size_mark=1.0, nb_test=200)
