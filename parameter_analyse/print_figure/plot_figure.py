import numpy as np
from scipy.io.matlab import loadmat
import os
from print_stability import print_stability
import matplotlib.pyplot as plt
from colorline import colorline
import matplotlib as mpl
import h5py


path = os.path.dirname(__file__) + '/../analyse_dynamic/'
H_LP = True
cmap = plt.get_cmap('jet')
norm = mpl.colors.Normalize(vmin=-100, vmax=100)
limit_circle = True

b_30 = loadmat(path + '/b_30/EQ_Low/EQ_Low.mat', chars_as_strings=True, simplify_cells=True)
b_30['x'][:2] *= 1000
b_30['x'][-1] *= 1000

b_60 = loadmat(path + '/b_60/EQ_Low/EQ_Low.mat', chars_as_strings=True, simplify_cells=True)
b_60['x'][:2] *= 1000
b_60['x'][-1] *= 1000

b_0 = loadmat(path + '/EQ_High/EQ_high.mat', chars_as_strings=True, simplify_cells=True)
b_0['x'][:2] *= 1000
b_0['x'][-1] *= 1000

if H_LP:
    H_high = loadmat(path + '/b_30/H/H_high.mat', chars_as_strings=True, simplify_cells=True)
    H_high['x'][:2] *= 1000
    H_high['x'][-3] *= 1000
    # H_low = loadmat(path + '/b_30/H/H_low_h.mat', chars_as_strings=True, simplify_cells=True)
    f = h5py.File(path + '/b_30/H/H_low_h.mat')
    H_low={'f': np.array(f['f']).swapaxes(0,1), 'h': np.array(f['h']).swapaxes(0,1), 'v': np.array(f['v']).swapaxes(0,1), 'x': np.array(f['x']).swapaxes(0,1)}
    H_low['s'] = [{'index': int(f[f['s']['index'][i][0]][0][0]),
                   'label': ''.join(chr(character) for character in f[f['s']['label'][i][0]]),
                   'msg': ''.join(chr(character) for character in f[f['s']['msg'][i][0]]),
                   } for i in range(f['s']['index'].shape[0])]
    H_low['x'][:2] *= 1000
    H_low['x'][-3] *= 1000

    LP_high = loadmat(path + '/b_30/LP/LP_high.mat', chars_as_strings=True, simplify_cells=True)
    LP_high['x'][:2] *= 1000
    LP_high['x'][-2] *= 1000
    # LP_middle_l = loadmat(path + '/b_30/LP/LP_middle_l.mat', chars_as_strings=True, simplify_cells=True)
    f = h5py.File(path + '/b_30/LP/LP_low_h.mat')
    LP_middle_l={'f': np.array(f['f']).swapaxes(0,1), 'h': np.array(f['h']).swapaxes(0,1), 'v': np.array(f['v']).swapaxes(0,1), 'x': np.array(f['x']).swapaxes(0,1)}
    LP_middle_l['s'] = [{'index': int(f[f['s']['index'][0][i]][0][0]),
                   'label': ''.join(chr(character) for character in f[f['s']['label'][0][i]]),
                   'msg': ''.join(chr(character) for character in f[f['s']['msg'][0][i]]),
                   } for i in range(f['s']['index'].shape[1])]
    LP_middle_l['x'][:2] *= 1000
    LP_middle_l['x'][-2] *= 1000


if limit_circle:
    limit_circle = loadmat(path + '/Limit_Circle/Limit_Circle.mat', chars_as_strings=True, simplify_cells=True)
    min = limit_circle['x'][:6]
    max = limit_circle['x'][:6]
    for i in range(1, int((limit_circle['x'].shape[0] - 2) / 6)):
        min = np.min([min, limit_circle['x'][i * 6:(i + 1) * 6]], axis=0)
        max = np.max([max, limit_circle['x'][i * 6:(i + 1) * 6]], axis=0)
    extra = limit_circle['x'][-2:]
    min[:2] *= 1000
    max[:2] *= 1000
    extra[1] *= 1000

fig = plt.figure()
line_b_0 = print_stability(b_0['x'], b_0['f'], b_0['s'], 6, 0, color='k')
line_b_30 = print_stability(b_30['x'], b_30['f'], b_30['s'], 6, 0, color='b')
line_b_60 = print_stability(b_60['x'], b_60['f'], b_60['s'], 6, 0, color='g')
if H_LP:
    for data in [H_high, H_low, LP_high, LP_middle_l]:
        colorline(data['x'][6, :], data['x'][0, :], data['x'][7, :], alpha=0.5, cmap=cmap, norm=norm)
        for points in data['s'][1:-1]:
            if points['msg'] != 'Zero-Hopf point: neutral saddle' and points['label'] != 'BV':
                plt.plot(data['x'][6, points['index'] - 1], data['x'][0, points['index'] - 1], 'mx', markersize=10.0)
                plt.text(data['x'][6, points['index'] - 1], data['x'][0, points['index'] - 1], "  " + points['label'])
                print(points['label'], data['x'][6, points['index'] - 1], data['x'][7, points['index'] - 1],
                      data['x'][0, points['index'] - 1], data['x'][1, points['index'] - 1])
if limit_circle:
    plt.plot(extra[1, :], max[0, :], '--', alpha=0.5, color='orange')
    plt.plot(extra[1, :], min[0, :], '--', alpha=0.5, color='orange')
plt.legend([line_b_60, line_b_30, line_b_0], ['b=60', 'b=30', 'b=0'])
plt.ylim(ymax=200.0, ymin=0.0)
plt.xlim(xmax=200.0, xmin=-30.0)
plt.xlabel("external input")
plt.ylabel("firing rate of excitatory population Hz")
cax = plt.axes([0.05, 0.1, 0.9, 0.02])
cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation='horizontal', label='Some Units')
cbar.set_label('adaptation')
plt.subplots_adjust(bottom=0.22, top=0.98)


plt.figure()
line_b_0 = print_stability(b_0['x'], b_0['f'], b_0['s'], 6, 1, color='k')
line_b_30 = print_stability(b_30['x'], b_30['f'], b_30['s'], 6, 1, color='b')
line_b_60 = print_stability(b_60['x'], b_60['f'], b_60['s'], 6, 1, color='g')
if H_LP:
    for data in [H_high, H_low, LP_high, LP_middle_l]:
        colorline(data['x'][6, :], data['x'][1, :], data['x'][7, :], alpha=0.5, cmap=cmap, norm=norm)
        for points in data['s'][1:-1]:
            if points['msg'] != 'Zero-Hopf point: neutral saddle' and points['label'] != 'BV':
                plt.plot(data['x'][6, points['index'] - 1], data['x'][1, points['index'] - 1], 'mx', markersize=10.0)
                plt.text(data['x'][6, points['index'] - 1], data['x'][1, points['index'] - 1], "  " + points['label'])
if limit_circle:
    plt.plot(extra[1, :], max[1, :], '--', alpha=0.5, color='orange')
    plt.plot(extra[1, :], min[1, :], '--', alpha=0.5, color='orange')
plt.legend([line_b_60, line_b_30, line_b_0], ['b=60', 'b=30', 'b=0'])
plt.ylim(ymax=200.0, ymin=0.0)
plt.xlim(xmax=200.0, xmin=-30.0)
plt.xlabel("external input")
plt.ylabel("firing rate of inhibitory population Hz")
cax = plt.axes([0.05, 0.1, 0.9, 0.02])
cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation='horizontal', label='Some Units')
cbar.set_label('adaptation')
plt.subplots_adjust(bottom=0.22, top=0.98)

plt.figure()
line_b_0 = print_stability(b_0['x'], b_0['f'], b_0['s'], 0, 1, color='k')
line_b_30 = print_stability(b_30['x'], b_30['f'], b_30['s'], 0, 1, color='b')
line_b_60 = print_stability(b_60['x'], b_60['f'], b_60['s'], 0, 1, color='g')
if H_LP:
    for data in [H_high, H_low, LP_high, LP_middle_l]:
        colorline(data['x'][0, :], data['x'][1, :], data['x'][7, :], alpha=0.5, cmap=cmap, norm=norm)
        for points in data['s'][1:-1]:
            if points['msg'] != 'Zero-Hopf point: neutral saddle' and points['label'] != 'BV':
                plt.plot(data['x'][0, points['index'] - 1], data['x'][1, points['index'] - 1], 'mx', markersize=10.0)
                plt.text(data['x'][0, points['index'] - 1], data['x'][1, points['index'] - 1], "  " + points['label'])
if limit_circle:
    plt.plot(max[0, :], max[1, :], '--', alpha=0.5, color='orange')
    plt.plot(min[0, :], min[1, :], '--', alpha=0.5, color='orange')
plt.legend([line_b_60, line_b_30, line_b_0], ['b=60', 'b=30', 'b=0'])
plt.ylim(ymax=200.0, ymin=0.0)
plt.xlim(xmax=200.0, xmin=0.0)
plt.xlabel("firing rate of excitatory population Hz")
plt.ylabel("firing rate of inhibitory population Hz")
cax = plt.axes([0.05, 0.1, 0.9, 0.02])
cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation='horizontal', label='Some Units')
cbar.set_label('adaptation')
plt.subplots_adjust(bottom=0.22, top=0.98)

plt.figure()
if H_LP:
    for data in [H_high, H_low, LP_high, LP_middle_l]:
        colorline(data['x'][7, :], data['x'][0, :], data['x'][6, :], alpha=0.5, cmap=cmap, norm=norm)
        for points in data['s'][1:-1]:
            if points['msg'] != 'Zero-Hopf point: neutral saddle' and points['label'] != 'BV':
                plt.plot(data['x'][7, points['index'] - 1], data['x'][0, points['index'] - 1], 'mx', markersize=10.0)
                plt.text(data['x'][7, points['index'] - 1], data['x'][0, points['index'] - 1], "  " + points['label'])
plt.ylim(ymax=200.0, ymin=0.0)
plt.xlim(xmax=200.0, xmin=-100.0)
plt.xlabel("adaptation")
plt.ylabel("firing rate of excitatory population Hz")
cax = plt.axes([0.05, 0.1, 0.9, 0.02])
cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation='horizontal', label='Some Units')
cbar.set_label('external input')
plt.subplots_adjust(bottom=0.22, top=0.98)

plt.figure()
if H_LP:
    for data in [H_high, H_low, LP_high, LP_middle_l]:
        colorline(data['x'][7, :], data['x'][1, :], data['x'][6, :], alpha=0.5, cmap=cmap, norm=norm)
        for points in data['s'][1:-1]:
            if points['msg'] != 'Zero-Hopf point: neutral saddle' and points['label'] != 'BV':
                plt.plot(data['x'][7, points['index'] - 1], data['x'][1, points['index'] - 1], 'mx', markersize=10.0)
                plt.text(data['x'][7, points['index'] - 1], data['x'][1, points['index'] - 1], "  " + points['label'])
plt.ylim(ymax=200.0, ymin=0.0)
plt.xlim(xmax=200.0, xmin=-100.0)
plt.xlabel("adaptation")
plt.ylabel("firing rate of inhibitory population Hz")
cax = plt.axes([0.05, 0.1, 0.9, 0.02])
cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation='horizontal', label='Some Units')
cbar.set_label('external input')
plt.subplots_adjust(bottom=0.22, top=0.98)
plt.show()
