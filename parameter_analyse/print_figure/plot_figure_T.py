from scipy.io.matlab import loadmat
import os
from print_stability import print_stability
import matplotlib.pyplot as plt
import numpy as np


path = os.path.dirname(__file__) + '/../analyse_dynamic/'

T_20 = loadmat(path + '/T_20/EQ_High/EQ_high.mat', chars_as_strings=True, simplify_cells=True)
T_20['x'][:2] *= 1000
T_20['x'][-1] *= 1000

T_50 = loadmat(path + '/T_50/EQ_High/EQ_high.mat', chars_as_strings=True, simplify_cells=True)
T_50['x'][:2] *= 1000
T_50['x'][-1] *= 1000

T_5 = loadmat(path + '/EQ_High/EQ_high.mat', chars_as_strings=True, simplify_cells=True)
T_5['x'][:2] *= 1000
T_5['x'][-1] *= 1000


fig = plt.figure()
line_T_5 = print_stability(T_5['x'], T_5['f'], T_5['s'], 6, 0, color='k')
line_T_20 = print_stability(T_20['x'], T_20['f'], T_20['s'], 6, 0, color='b')
line_T_50 = print_stability(T_50['x'], T_50['f'], T_50['s'], 6, 0, color='g')
plt.legend([line_T_50, line_T_20, line_T_5], ['T=50', 'T=20', 'T=5'])
plt.ylim(ymax=200.0, ymin=0.0)
plt.xlim(xmax=200.0, xmin=-30.0)
plt.xlabel("external input")
plt.ylabel("firing rate of excitatory population Hz")
cax = plt.axes([0.05, 0.1, 0.9, 0.02])
plt.subplots_adjust(bottom=0.22, top=0.98)


plt.figure()
line_T_5 = print_stability(T_5['x'], T_5['f'], T_5['s'], 6, 1, color='k')
line_T_20 = print_stability(T_20['x'], T_20['f'], T_20['s'], 6, 1, color='b')
line_T_50 = print_stability(T_50['x'], T_50['f'], T_50['s'], 6, 1, color='g')
plt.legend([line_T_50, line_T_20, line_T_5], ['T=50', 'T=20', 'T=5'])
plt.ylim(ymax=200.0, ymin=0.0)
plt.xlim(xmax=200.0, xmin=-30.0)
plt.xlabel("external input")
plt.ylabel("firing rate of inhibitory population Hz")
cax = plt.axes([0.05, 0.1, 0.9, 0.02])
plt.subplots_adjust(bottom=0.22, top=0.98)

plt.figure()
line_T_5 = print_stability(T_5['x'], T_5['f'], T_5['s'], 0, 1, color='k')
line_T_20 = print_stability(T_20['x'], T_20['f'], T_20['s'], 0, 1, color='b')
line_T_50 = print_stability(T_50['x'], T_50['f'], T_50['s'], 0, 1, color='g')
plt.legend([line_T_50, line_T_20, line_T_5], ['T=50', 'T=20', 'T=5'])
plt.ylim(ymax=200.0, ymin=0.0)
plt.xlim(xmax=200.0, xmin=0.0)
plt.xlabel("firing rate of excitatory population Hz")
plt.ylabel("firing rate of inhibitory population Hz")
plt.subplots_adjust(bottom=0.22, top=0.98)

plt.figure()
for i in range(6):
    plt.plot(np.real(T_5['f'][i]), np.imag(T_5['f'][i]),'.')
plt.figure()
for i in range(6):
    plt.plot(np.real(T_20['f'][i]), np.imag(T_20['f'][i]),'.')
plt.figure()
for i in range(6):
    plt.plot(np.real(T_50['f'][i]), np.imag(T_50['f'][i]),'.')

for i in range(3):
    plt.figure()
    plt.plot(T_5['h'][-i],'.',label='T_5')
    plt.plot(T_20['h'][-i],'.',label='T_20')
    plt.plot(T_50['h'][-i],'.',label='T_50')
    plt.legend()
plt.show()
