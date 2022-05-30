import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
def split_list(a_list):
    i_half = len(a_list) // 2
    return a_list[:i_half], a_list[i_half:]
markers = [m for m, func in Line2D.markers.items()
           if func != 'nothing' and m not in Line2D.filled_markers]

from scipy import stats
from statsmodels.compat.python import lzip

from function_autocorrelation import autocorr, autocorr1, autocorr2, autocorr3, autocorr4, autocorr5
from statsmodels.graphics.tsaplots import plot_acf
from elephant.spike_train_correlation import spike_train_timescale, cross_correlation_histogram, \
    correlation_coefficient, covariance
from elephant.conversion import BinnedSpikeTrain
import quantities as pq

from get_data import get_hist, get_spiketrains

# https://stackoverflow.com/questions/643699/how-can-i-use-numpy-correlate-to-do-autocorrelation
# https://www.researchgate.net/profile/Eran-Stark/publication/5414094_Dependence_of_Neuronal_Correlations_on_Filter_Characteristics_and_Marginal_Spike_Train_Statistics/links/0912f51247a3f75ea5000000/Dependence-of-Neuronal-Correlations-on-Filter-Characteristics-and-Marginal-Spike-Train-Statistics.pdf?origin=figuresDialog_download
# https://elephant.readthedocs.io/en/latest/reference/_toctree/spike_train_correlation/elephant.spike_train_correlation.spike_train_timescale.html#elephant.spike_train_correlation.spike_train_timescale

nest = True
# path='/home/kusch/Documents/project/co_simulation/TVB-NEST-proof/result_sim/nest_only/nest/'
# path = '/home/kusch/Documents/project/co_simulation/TVB-NEST-proof/result_sim/co-simulation/nest/'
path = '/home/kusch/Documents/project/co_simulation/TVB-NEST-proof/result_sim/co-simulation_neurolib/nest/'
nb_neuron = 50
# nest = False
# path = "/home/kusch/Documents/project/co_simulation/TVB-NEST-proof/result_sim/co-simulation_neuron/neuron/data.npy"
# # path = "/home/kusch/Documents/project/co_simulation/TVB-NEST-proof/result_sim/co-simulation_neurolib_neuron/neuron/data.npy"
# nb_neuron = 100
begin = 200.0  # in ms
lag = 50
end = 1001.0  # in ms
binsize = 1  # in ms
alpha = 0.05
data = get_hist(path, hist_binwidth=binsize,begin=begin, end=end, nest=nest)
spiketrains, bsts = get_spiketrains(path, nb_neurons=nb_neuron, begin=begin, end=end, binwidth=binsize, nest=nest)

plt.figure()
plt.plot(data)

plot_acf(data, lags=lag)

plt.figure()
for index, autocorr_f in enumerate([autocorr, autocorr1, autocorr2, autocorr3, autocorr4, autocorr5]):
    plt.plot(autocorr_f(data, range(lag)), marker=markers[index % len(markers)], label=str(index))

bin_hist = BinnedSpikeTrain(np.expand_dims(np.array(data.tolist()), 0), t_start=begin * pq.ms, t_stop=end * pq.ms, bin_size=binsize* pq.ms)
cc_hist, lags = cross_correlation_histogram(bin_hist, bin_hist, window=[-lag, lag], cross_correlation_coefficient=True)
index_lag = np.where(lags>=0)
plt.plot(lags[index_lag], np.array(cc_hist)[index_lag], marker=markers[(index+1) % len(markers)], label=str(index+1))
print(spike_train_timescale(bin_hist, max_tau=np.max(lag) * pq.ms))
plt.legend()

nobs = len(data)  # TODO: should this shrink for missing="drop" and NaNs in x?
acf = np.array(cc_hist).squeeze(1)[index_lag]
varacf = np.ones_like(acf) / nobs
varacf[0] = 0
varacf[1] = 1.0 / nobs
varacf[2:] *= 1 + 2 * np.cumsum(acf[1:-1] ** 2)
interval = stats.norm.ppf(1.0 - alpha / 2.0) * np.sqrt(varacf)
confint = np.array(lzip(acf - interval, acf + interval))
confint[0] = acf[0]  # fix confidence interval for lag 0 to varpacf=0
lags = lags.astype(float)
lags[index_lag][0] -= 0.5
lags[-1] += 0.5
plt.fill_between(lags[index_lag][1:],
                 np.array(confint[1:, 0] - acf[1:]),
                 np.array(confint[1:, 1] - acf[1:]), alpha=0.25)



plt.figure()
timescale = []
for bst in bsts:
    timescale.append(spike_train_timescale(bst, max_tau=np.max(lag) * pq.ms))
    cc_hist, lags = cross_correlation_histogram(
        bst, bst, window=[-lag, lag], cross_correlation_coefficient=True)
    plt.plot(lags, cc_hist)
print(timescale)
print(np.mean(timescale), np.std(timescale), np.max(timescale), np.min(timescale))



corrcoef = correlation_coefficient(bsts)
plt.figure()
np.fill_diagonal(corrcoef,0.0)
plt.imshow(corrcoef)
plt.colorbar()

covar = covariance(bsts)
plt.figure()
plt.imshow(covar)
plt.colorbar()

plt.show()
