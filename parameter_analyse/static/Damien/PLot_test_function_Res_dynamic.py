import numpy as np
import matplotlib.pyplot as plt

plt.rc('xtick', labelsize=15)
plt.rc('ytick', labelsize=15)

Cnt = 0
# Lb=[i*2 for i in range(60)]
# Lei=[i*0.1 for i in range(20)]
Lb = [0, 30, 60]
Lei = [i for i in range(100)]
seeds = list(range(50))

result = {'fr_ex': [],
          'fr_in': [],
          'std_ex': [],
          'std_in': [],
          'cov': []
          }
for bVal in Lb:
    for key in result.keys():
        result[key].append([])
    for ExIn in Lei:
        for key in result.keys():
            result[key][-1].append([])
        for seed in seeds:
            print('Results3/AD_popRateInh_ExIn_' + str(ExIn) + '_bval_' + str(bVal) + 'Nseed_' + str(seed) + '.npy')
            FRinh = np.load(
                'Results3/AD_popRateInh_ExIn_' + str(ExIn) + '_bval_' + str(bVal) + 'Nseed_' + str(seed) + '.npy')
            FRexc = np.load(
                'Results3/AD_popRateExc_ExIn_' + str(ExIn) + '_bval_' + str(bVal) + 'Nseed_' + str(seed) + '.npy')
            result['fr_ex'][-1][-1].append(np.mean(FRexc))
            result['fr_in'][-1][-1].append(np.mean(FRinh))
            cov = np.cov([FRinh, FRexc])
            result['std_ex'][-1][-1].append(cov[0, 0])
            result['std_in'][-1][-1].append(cov[1, 1])
            result['cov'][-1][-1].append(cov[0, 1])

for key in result.keys():
    result[key] = np.array(result[key])
for index, bVal in enumerate(Lb):
    plt.figure()
    plt.plot(Lei, np.mean(result['fr_ex'][index], axis=1), '.')
    plt.violinplot(np.swapaxes(result['fr_ex'][index], 0, 1), positions=Lei, showmeans=True, showextrema=True,
                   showmedians=True)
    plt.title(str(bVal) + ' firing rate exc')
    plt.figure()
    plt.plot(Lei, np.mean(result['fr_in'][index], axis=1), '.')
    plt.violinplot(np.swapaxes(result['fr_in'][index], 0, 1), positions=Lei, showmeans=True, showextrema=True,
                   showmedians=True)
    plt.title(str(bVal) + ' firing rate exc')
    plt.figure()
    plt.plot(Lei, np.mean(result['std_ex'][index], axis=1), '.')
    plt.title(str(bVal) + ' firing rate exc')
    plt.figure()
    plt.plot(Lei, np.mean(result['std_in'][index], axis=1), '.')
    plt.title(str(bVal) + ' firing rate exc')
    plt.figure()
    plt.plot(Lei, np.mean(result['cov'][index], axis=1), '.')
    plt.title(str(bVal) + ' firing rate exc')

plt.figure()
plt.plot(Lei, np.mean(result['fr_ex'][0], axis=1), '.')
plt.plot(Lei, np.mean(result['fr_ex'][1], axis=1), '.')
plt.plot(Lei, np.mean(result['fr_ex'][2], axis=1), '.')


plt.figure()
FRexc_0 = np.load('Results3/AD_popRateExc_ExIn_10_bval_0Nseed_10.npy')
FRexc_30 = np.load('Results3/AD_popRateExc_ExIn_10_bval_30Nseed_10.npy')
FRexc_60 = np.load('Results3/AD_popRateExc_ExIn_10_bval_60Nseed_10.npy')
plt.plot(FRexc_0, '-')
plt.plot(FRexc_30, '-.')
plt.plot(FRexc_60, '--')
plt.show()
