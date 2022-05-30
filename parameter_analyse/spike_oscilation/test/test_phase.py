from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import math

# calculate PLV from 2 vectors of phases
def PLV(theta1, theta2):
  complex_phase_diff = np.exp(1j*(np.unwrap(theta1 - theta2)))
  plv = np.abs(np.sum(complex_phase_diff))/len(theta1)
  angle = np.angle(np.sum(complex_phase_diff)/len(theta1))
  angle_2 = np.angle(np.mean(np.cos(theta1 - theta2))+1j*np.mean(np.sin(theta1-theta2)))
  return plv, angle, math.degrees(angle), angle_2

frequency = 1
# Input signal
I_signal = np.sin(2*np.pi*frequency*np.arange(0, 20000, 1.0)*1e-3)
I_theta = np.angle(signal.hilbert(I_signal))

# histogram
hist_1 = np.load('hist_1_1.npy', allow_pickle=True)[0]
hist_norm = hist_1/(np.max(hist_1)-np.min(hist_1))*2-1

# filtering
b, a = signal.butter(1, [0.9, 1.1], fs=1e3, btype='band')
# b, a = signal.butter(1, 1.0, fs=1e3, btype='low')
hist_filter = signal.lfilter(b, a, hist_1)
# hist_filter = signal.lfilter(b, a, hist_norm)
hist_filter_norm = hist_filter/(np.max(hist_filter)-np.min(hist_filter))*2

# get phases using Hilbert transofrm
theta = np.angle(signal.hilbert(hist_1))
theta_norm = np.angle(signal.hilbert(hist_norm))
theta_filter = np.angle(signal.hilbert(hist_filter))
theta_filter_norm = np.angle(signal.hilbert(hist_filter_norm))

# remove beginning
hist_1 = hist_1[5000:]
hist_norm = hist_norm[5000:]
hist_filter = hist_filter[5000:]
hist_filter_norm = hist_filter_norm[5000:]
theta = theta[5000:]
theta_norm = theta_norm[5000:]
theta_filter = theta_filter[5000:]
theta_filter_norm = theta_filter_norm[5000:]
I_signal = I_signal[5000:]
I_theta = I_theta[5000:]

# PLV
print(PLV(theta, I_theta))
print(PLV(theta_norm, I_theta))
print(PLV(theta_filter, I_theta))
print(PLV(theta_filter_norm, I_theta))

# plot signal normalize
ax1 = plt.subplot(221)
ax1.plot(hist_norm)
ax1.plot()
ax1.plot(I_signal)
ax2 = plt.subplot(222)
ax2.plot(hist_1/(np.max(hist_1)-np.min(hist_1))*2-1)
ax3 = plt.subplot(223)
ax3.plot(hist_filter/(np.max(hist_filter)-np.min(hist_filter))*2)
ax3.set_title('hist filter')
ax4 = plt.subplot(224)
ax4.plot(I_signal)

# plot
plt.figure()
ax1 = plt.subplot(221)
ax1.plot(theta)
ax1.plot(theta_filter)
ax1.plot(theta_norm)
ax1.plot(I_theta)
ax2 = plt.subplot(222)
ax2.plot(theta)
ax3 = plt.subplot(223)
ax3.plot(theta_filter)
ax4 = plt.subplot(224)
ax4.plot(theta_norm)

plt.figure()
ax1 = plt.subplot(231)
ax1.plot(theta-I_theta, '.', markersize=1, label='diff signal')
ax1.plot(theta_filter-I_theta, '.', markersize=1, label='diff filter')
ax1.plot(theta_norm-I_theta, '.', markersize=1, label='diff signal norm')
ax1.plot(theta_filter_norm-I_theta, '.', markersize=1, label= 'diff filter norm')
ax1.legend()
ax2 = plt.subplot(232)
ax2.plot(theta-I_theta, '.')
ax3 = plt.subplot(233)
ax3.plot(theta_filter-I_theta, '.')
ax4 = plt.subplot(234)
ax4.plot(theta_norm-I_theta, '.')
ax5 = plt.subplot(235)
ax5.plot(theta_filter_norm-I_theta, '.')
plt.show()


def filter(hist_1, I_theta, frequency, low_frequency, high_frequency, order=5):
  # filtering
  b, a = signal.butter(order, [low_frequency, high_frequency], fs=1e3, btype='band')
  hist_filter = signal.lfilter(b, a, hist_1)
  theta_filter = np.angle(signal.hilbert(hist_filter))
  hist_filter = hist_filter[5000:]
  theta_filter = theta_filter[5000:]
  print('low frequency',low_frequency, 'high frequecny', high_frequency, ' PLV filter ', PLV(theta_filter, I_theta),
        'Mean phase shift', np.angle(np.mean(np.cos(theta_filter-I_theta))+1j*np.mean(np.sin(theta_filter-I_theta)))
        )

  plt.plot(hist_filter, label = str(low_frequency)+" "+str(high_frequency))
  # plt.plot(hist_1, label = 'all')
  # plt.plot(theta_filter-I_theta, '.', label = str(low_frequency)+" "+str(high_frequency))
  # plt.plot(np.real(np.exp(1j*(np.unwrap(theta_filter - I_theta)))), '.', label = str(low_frequency)+" "+str(high_frequency), markersize=1)
  # plt.plot(np.imag(np.exp(1j*(np.unwrap(theta_filter - I_theta)))), '.', label = str(low_frequency)+" "+str(high_frequency), markersize=1)
  # plt.plot(np.abs(np.exp(1j*(np.unwrap(theta_filter - I_theta)))), '.', label = str(low_frequency)+" "+str(high_frequency), markersize=1)
  # plt.plot(np.imag(np.exp(1j*(np.unwrap(theta_filter - I_theta)))),
  #          np.real(np.exp(1j*(np.unwrap(theta_filter - I_theta)))),
  #          '.', label = str(low_frequency)+" "+str(high_frequency), markersize=1)


hist_1 = np.load('hist_1_1.npy', allow_pickle=True)[0]
f, Pxx_den =signal.welch(hist_1, fs=1e3, nperseg=1e3)
plt.semilogy(f, Pxx_den)

plt.figure()
plt.plot(hist_1/10)

# plt.figure()
# for i in range(2, 23):
#     filter(hist_1, I_theta, frequency, low_frequency=i, high_frequency=30, order=5)
# plt.legend()
# plt.figure()
# for i in range(27, 70, 3):
#     filter(hist_1, I_theta, frequency, low_frequency=20, high_frequency=i, order=5)
# plt.legend()
# plt.show()
# hist_norm = hist_1/(np.max(hist_1)-np.min(hist_1))*2-1
# plt.figure()
# for i in range(2,23):
#    filter(hist_norm, I_theta, frequency, low_frequency=i,high_frequency=30, order=5)
# plt.legend()
plt.show()


# import numpy as np
# from sklearn.decomposition import FastICA, PCA
# pca = PCA(n_components=1)
# S_ = pca.fit_transform(np.expand_dims(hist_1,1))  # Reconstruct signals
# plt.plot(S_.T)
filter(hist_1[5000:], I_theta[5000:], frequency, low_frequency=24, high_frequency=26, order=5)
plt.show()
