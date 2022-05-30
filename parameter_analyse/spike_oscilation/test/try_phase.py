from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
from elephant.spectral import welch_psd

# calculate PLV from 2 vectors of phases
def PLV(theta1, theta2):
  complex_phase_diff = np.exp(1j*(np.unwrap(theta1 - theta2)))
  plv = np.abs(np.sum(complex_phase_diff))/len(theta1)
  return plv

def filter(hist_1, I_theta, frequency, low_frequency, high_frequency, order=1, remove_start_hilbert=2500, remove_end=2500):
  # filtering
  b, a = signal.butter(order, [low_frequency, high_frequency], fs=1e3, btype='band')
  hist_filter = signal.lfilter(b, a, hist_1)
  theta_filter = np.angle(signal.hilbert(hist_filter))
  hist_filter = hist_filter[remove_start_hilbert:-remove_end]
  theta_filter = theta_filter[remove_start_hilbert:-remove_end]
  print('low frequency',low_frequency, 'high frequecny', high_frequency, ' PLV filter ', PLV(theta_filter, I_theta),
        'Mean phase shift', np.angle(np.mean(np.cos(I_theta - theta_filter))+1j*np.mean(np.sin(I_theta-theta_filter)))
        )
  return hist_filter, theta_filter


def frequency_explore(frequency, hist_path, remove_init=500, remove_start_hilbert=3000, remove_end=2500, begin=0.0,
                      end=20000.0, fs=1e3):

  # Input signal
  I_signal = np.sin(2*np.pi*frequency*np.arange(begin, end, fs*1e-3)*1e-3)
  I_theta = np.angle(signal.hilbert(I_signal))


  hist_1 = np.load(hist_path, allow_pickle=True)[0][remove_init:]
  welch_hist_1 = welch_psd(hist_1, frequency_resolution=1.0, fs=fs, )
  if welch_hist_1[0][np.argmax(welch_hist_1[1])] != frequency:
    print(welch_hist_1[0][np.argmax(welch_hist_1[1])])
    raise Exception('bad frequency')


  hist_filter, theta_filter = filter(hist_1, I_theta[remove_start_hilbert+remove_init:-remove_end], frequency,
                                     low_frequency=frequency-0.1, high_frequency=frequency+0.1, order=1,
                                     remove_start_hilbert=remove_start_hilbert, remove_end=remove_end)

  # plt.figure()
  # plt.plot(hist_1)
  # plt.title(str(frequency))
  # plt.figure()
  # plt.semilogy(welch_hist_1[0], welch_hist_1[1])
  # plt.title(str(frequency))
  # plt.figure()
  # plt.plot(hist_filter/(np.max(hist_filter)-np.min(hist_filter))*2)
  # plt.plot(I_signal[remove_start_hilbert+remove_init:-remove_end], alpha=0.5)
  # plt.title('histogram')
  plt.figure()
  plt.subplot(311)
  plt.plot(hist_1[remove_start_hilbert:-remove_end]/(np.max(hist_1[remove_start_hilbert:-remove_end])-np.min(hist_1[remove_start_hilbert:-remove_end]))*2-1, color='blue')
  plt.plot(hist_filter/(np.max(hist_filter)-np.min(hist_filter))*2, '--', color='orange')
  plt.plot(I_signal[remove_start_hilbert+remove_init:-remove_end], alpha=0.5, color='green')
  plt.title('angle')
  plt.subplot(312)
  plt.plot(theta_filter, color='orange')
  plt.plot(I_theta[remove_start_hilbert+remove_init:-remove_end], alpha=0.5,color='green')
  plt.title('angle')
  plt.subplot(313)
  plt.plot(theta_filter-I_theta[remove_start_hilbert+remove_init:-remove_end], '.')
  plt.title('difference')


# for i in [1,5,10,15,20,25]:
#   hist_path= 'hist_1_'+str(i)+'.npy'
#   frequency_explore(i, hist_path)

# for i in [1,5,10,15,20,25,30]:
#   hist_path= 'hist_1_'+str(i)+'_400.npy'
#   frequency_explore(i, hist_path)
frequency_explore(30.0, 'hist_1_30_400.npy')
plt.show()