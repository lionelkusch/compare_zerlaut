#  Copyright 2023 Aix-Marseille Université
# "Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements; and to You under the Apache License, Version 2.0. "
import os
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm


# hypothesis
# v = 0 => c_ee = 0 and c_ei = 0
# T dv/dt = (TF - v) + 0.5 * c_e_e d2TF_e/dee + 1 * c_e_i + 0.5*c_ii * d2TF_e/dii
# 0 = TF + 0.5 * c_i_i
# c_ii - 2 * TF
def plot_values(fig, ax, data, ranges_x, ranges_y, max_cut, cmap, min_cut, ticks_size, colorbar_add=False):
    """
    plot value in log scale
    :param fig: figure
    :param ax: axis
    :param data: data to plot
    :param ranges_x: range of x
    :param ranges_y: range of y
    :param max_cut: maximum of cut
    :param cmap: color map
    :param min_cut: minimum cut
    :return:
    """
    max_limit = np.nanmax(data) if (min_cut < np.nanmax(data) < max_cut) else max_cut
    min_limit = np.nanmin(data) if np.nanmin(data) > min_cut else min_cut
    im_0 = ax.imshow(data, cmap=cmap, norm=LogNorm(vmin=min_limit, vmax=max_limit))
    # im_0 = ax.imshow(data, cmap=cmap, norm=SymLogNorm(linthresh=min_limit, vmin=-max_limit, vmax=max_limit))
    if colorbar_add:
        colorbar = fig.colorbar(im_0, ax=ax)
        colorbar.ax.tick_params(labelsize=ticks_size)
        colorbar.ax.set_ylabel('minimal variance', fontdict={'fontsize': labelticks_size})
    ticks_positions = np.array(np.around(np.linspace(0, ranges_x.shape[0] - 1, 4)), dtype=int)
    ax.set_yticks(ticks_positions)
    ax.set_yticklabels(np.around(ranges_x[ticks_positions], decimals=3) * 1e3)
    ticks_positions = np.array(np.around(np.linspace(0, ranges_y.shape[0] - 1, 4)), dtype=int)
    ax.set_xticks(ticks_positions)
    ax.set_xticklabels(np.around(ranges_y[ticks_positions], decimals=0))
    if np.nanmin(data) < 0.0:
        ax.contourf(np.arange(0, data.shape[0], 1), np.arange(0, data.shape[1], 1),
                    data[:, :, 0], levels=[-10000.0, 0.0], hatches=['/'], colors=['red'], alpha=0.0)
        ax.contour(np.arange(0, data.shape[0], 1), np.arange(0, data.shape[1], 1),
                   data[:, :, 0], levels=[0.0], colors=['black'], linewidths=0.1)
    ax.invert_yaxis()

# parameters of for getting data and plotting
path = os.path.dirname(os.path.realpath(__file__)) + '/../../check_derivation/data/'
min_cut = 1e-8
max_cut = 1e50
labelticks_size = 12
label_legend_size = 12
ticks_size = 12
cmap = cm.get_cmap('viridis')
ranges_excitatory = [np.linspace(0.0, 10.0, 1000) * 1e-3, np.linspace(0.0, 200.0, 1000) * 1e-3]
ranges_inhibitory = [np.linspace(0.0, 10.0, 1000) * 1e-3, np.linspace(0.0, 200.0, 1000) * 1e-3]
ranges_adaptation = [np.linspace(0.0, 100.0, 1000), np.linspace(0.0, 10000.0, 1000)]
## Get data
# excitatory transfer function
i_inh = 1
i_inh_adp = 1
ex_TF = np.load(path + "exc_" + str(i_inh) + "_" + str(i_inh_adp) + "_transfer_function.npy")
ex_diff2_fe_fe = np.load(path + "exc_" + str(i_inh) + "_" + str(i_inh_adp) + "_second_order_derivation_fe.npy")
ex_diff2_fi_fi = np.load(path + "exc_" + str(i_inh) + "_" + str(i_inh_adp) + "_second_order_derivation_fi.npy")
ex_diff2_fe_fi = np.load(path + "exc_" + str(i_inh) + "_" + str(i_inh_adp) + "_second_order_derivation_fe_and_fi.npy")
i_zoom_inh = 0
i_zoom_inh_adp = 1
ex_zoom_TF = np.load(path + "exc_" + str(i_zoom_inh) + "_" + str(i_zoom_inh_adp) + "_transfer_function.npy")
ex_zoom_diff2_fe_fe = np.load(path + "exc_" + str(i_zoom_inh) + "_" + str(i_zoom_inh_adp) + "_second_order_derivation_fe.npy")
ex_zoom_diff2_fi_fi = np.load(path + "exc_" + str(i_zoom_inh) + "_" + str(i_zoom_inh_adp) + "_second_order_derivation_fi.npy")
ex_zoom_diff2_fe_fi = np.load(path + "exc_" + str(i_zoom_inh) + "_" + str(i_zoom_inh_adp) + "_second_order_derivation_fe_and_fi.npy")

# inhibitory transfer function
i_ex = 1
i_ex_adp = 1
in_TF = np.load(path + "inh_" + str(i_ex) + "_" + str(i_ex_adp) + "_transfer_function.npy")
in_diff2_fe_fe = np.load(path + "inh_" + str(i_ex) + "_" + str(i_ex_adp) + "_second_order_derivation_fe.npy")
in_diff2_fi_fi = np.load(path + "inh_" + str(i_ex) + "_" + str(i_ex_adp) + "_second_order_derivation_fi.npy")
in_diff2_fe_fi = np.load(path + "inh_" + str(i_ex) + "_" + str(i_ex_adp) + "_second_order_derivation_fe_and_fi.npy")
i_zoom_ex = 0
i_zoom_ex_adp = 1
in_zoom_TF = np.load(path + "inh_" + str(i_zoom_ex) + "_" + str(i_zoom_ex_adp) + "_transfer_function.npy")
in_zoom_diff2_fe_fe = np.load(path + "inh_" + str(i_zoom_ex) + "_" + str(i_zoom_ex_adp) + "_second_order_derivation_fe.npy")
in_zoom_diff2_fi_fi = np.load(path + "inh_" + str(i_zoom_ex) + "_" + str(i_zoom_ex_adp) + "_second_order_derivation_fi.npy")
in_zoom_diff2_fe_fi = np.load(path + "inh_" + str(i_zoom_ex) + "_" + str(i_zoom_ex_adp) + "_second_order_derivation_fe_and_fi.npy")

## plot figure
fig, axs = plt.subplots(2, 2, figsize=(6.8, 5.5), gridspec_kw={'width_ratios': [0.9, 1.1]})
ex_diff_value = - ex_TF * 2 / ex_diff2_fi_fi * 1e6
ex_diff_value[np.where(ex_diff_value <= 0.)] = np.NAN
plot_values(fig, axs[0, 0], ex_diff_value, ranges_inhibitory[i_inh], ranges_adaptation[i_inh_adp], max_cut, cmap, min_cut, ticks_size)
axs[0, 0].set_ylabel("firing rate of inhibitory\npopulation (Hz)", fontdict={'fontsize': labelticks_size})
axs[0, 0].tick_params(labelsize=ticks_size)
axs[0, 0].annotate('A', xy=(-0.2, 0.90), xycoords='axes fraction', weight='bold', fontsize=labelticks_size)
in_diff_value = - in_TF * 2 / in_diff2_fe_fe * 1e6
in_diff_value[np.where(in_diff_value <= 0.)] = np.NAN
plot_values(fig, axs[1, 0], in_diff_value, ranges_excitatory[i_ex], ranges_adaptation[i_ex_adp], max_cut, cmap, min_cut, ticks_size)
axs[1, 0].set_ylabel("firing rate of excitatory\npopulation (Hz)", fontdict={'fontsize': labelticks_size})
axs[1, 0].set_xlabel("adaptation", fontdict={'fontsize': labelticks_size})
axs[1, 0].tick_params(labelsize=ticks_size)
axs[1, 0].annotate('B', xy=(-0.2, 0.90), xycoords='axes fraction', weight='bold', fontsize=labelticks_size)
# zoom
ex_zoom_diff_value = - ex_zoom_TF * 2 / ex_zoom_diff2_fi_fi * 1e6
ex_zoom_diff_value[np.where(ex_zoom_diff_value <= 0.)] = np.NAN
plot_values(fig, axs[0, 1], ex_zoom_diff_value, ranges_inhibitory[i_zoom_inh], ranges_adaptation[i_zoom_inh_adp], max_cut, cmap, min_cut, ticks_size, colorbar_add=True)
axs[0, 1].tick_params(labelsize=ticks_size)
axs[0, 1].annotate('C', xy=(-0.2, 0.90), xycoords='axes fraction', weight='bold', fontsize=labelticks_size)
in_zoom_diff_value = - in_zoom_TF * 2 / in_zoom_diff2_fe_fe * 1e6
in_zoom_diff_value[np.where(in_zoom_diff_value <= 0.)] = np.NAN
plot_values(fig, axs[1, 1], in_zoom_diff_value, ranges_excitatory[i_zoom_ex], ranges_adaptation[i_zoom_ex_adp], max_cut, cmap, min_cut, ticks_size, colorbar_add=True)
axs[1, 1].set_xlabel("adaptation", fontdict={'fontsize': labelticks_size})
axs[1, 1].tick_params(labelsize=ticks_size)
axs[1, 1].annotate('D', xy=(-0.2, 0.90), xycoords='axes fraction', weight='bold', fontsize=labelticks_size)

# plt.suptitle('Minimum of variance for negative derivation', fontsize=labelticks_size)
plt.subplots_adjust(top=0.97, bottom=0.090, left=0.16, right=0.94, hspace=0.15, wspace=0.20)

plt.savefig('./figure/SP_figure_8.png', dpi=300)
# plt.show()