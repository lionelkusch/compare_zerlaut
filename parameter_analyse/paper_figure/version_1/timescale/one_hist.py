import numpy as np
import matplotlib.pyplot as plt
from parameter_analyse.static.python_file.analysis.analysis_global import time_scale, get_gids, load_spike, decay_exponential


def timescale_one_hist(path, begin=1000.0, end=5000.0, number=0, dt=0.1):
    duration = end-begin
    gids = get_gids(path, number)
    data_all = load_spike(gids, path, begin, end, number)
    hist_ex = np.histogram(data_all[:, 1], bins=int((end - begin) / dt))
    plt.figure()
    plt.plot(np.arange(0, duration, dt), hist_ex[0])

    coefficients, time_lag, popt = time_scale(hist=hist_ex, dt=dt, duration=duration)
    plt.figure()
    plt.plot(time_lag*dt, coefficients)
    # plt.plot(time_lag[peaks], signal[peaks], '.')
    plt.plot(time_lag*dt, decay_exponential(time_lag*dt, popt))
    print("decay :"+str(popt)+' '+str(dt))


if __name__ == '__main__':
    import os
    path = os.path.dirname(os.path.realpath(__file__)) + "/../../../static/"
    timescale_one_hist(path + "/simulation/data/master_seed_0/_b_0.0_rate_10.0/", dt=0.1)
    timescale_one_hist(path + "/simulation/data/master_seed_0/_b_0.0_rate_10.0/", dt=1.0)
    plt.show()
