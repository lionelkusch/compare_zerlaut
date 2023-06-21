#  Copyright 2023 Aix-Marseille Université
# "Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements; and to You under the Apache License, Version 2.0. "
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from parameter_analyse.fitting_procedure.Zerlaut import ZerlautAdaptationSecondOrder
from matplotlib.colors import SymLogNorm, Normalize


def model(parameter, path_TF_e, path_TF_i):
    """
    Parametrize the mean field model
    :param parameter: parameters of the neurons
    :param path_TF_e: polynomial of excitatory neurons
    :param path_TF_i: polynomial of inhibitory neurons
    :return: mean field model fitted
    """
    model_test = ZerlautAdaptationSecondOrder()
    model_test.g_L = np.array(parameter['g_L'])
    model_test.E_L_e = np.array(parameter['E_L_e'])
    model_test.E_L_i = np.array(parameter['E_L_i'])
    model_test.C_m = np.array(parameter['C_m'])
    model_test.a_e = np.array(parameter['a'])
    model_test.b_e = np.array(parameter['b'])
    model_test.a_i = np.array(parameter['a'])
    model_test.b_i = np.array(parameter['b'])
    model_test.tau_w_e = np.array(parameter['tau_w_e'])
    model_test.tau_w_i = np.array(parameter['tau_w_i'])
    model_test.E_e = np.array(parameter['E_ex'])
    model_test.E_i = np.array(parameter['E_in'])
    model_test.Q_e = np.array(parameter['Q_e'])
    model_test.Q_i = np.array(parameter['Q_i'])
    model_test.tau_e = np.array(parameter['tau_syn_ex'])
    model_test.tau_i = np.array(parameter['tau_syn_in'])
    model_test.N_tot = np.array(parameter['N_tot'])
    model_test.p_connect_e = np.array(parameter['p_connect_ex'])
    model_test.p_connect_i = np.array(parameter['p_connect_in'])
    model_test.g = np.array(parameter['g'])
    model_test.T = np.array(parameter['t_ref'])
    model_test.external_input_in_in = np.array(0.0)
    model_test.external_input_in_ex = np.array(0.0)
    model_test.external_input_ex_in = np.array(0.0)
    model_test.external_input_ex_ex = np.array(0.0)
    model_test.K_ext_e = np.array(0)
    model_test.K_ext_i = np.array(0)
    model_test.P_e = np.load(path_TF_e)
    model_test.P_i = np.load(path_TF_i)

    # Derivatives taken numerically : use a central difference formula with spacing `dx`
    def _diff_fe(TF, fe, fi, fe_ext, fi_ext, W, df=1e-10):
        return (TF(fe + df, fi, fe_ext, fi_ext, W) - TF(fe - df, fi, fe_ext, fi_ext, W)) / (2 * df * 1e3)

    model_test._diff_fe = _diff_fe

    def _diff_fi(TF, fe, fi, fe_ext, fi_ext, W, df=1e-10):
        return (TF(fe, fi + df, fe_ext, fi_ext, W) - TF(fe, fi - df, fe_ext, fi_ext, W)) / (2 * df * 1e3)

    model_test._diff_fi = _diff_fi

    def _diff2_fe_fe_e(fe, fi, fe_ext, fi_ext, W, df=1e-10):
        TF = model_test.TF_excitatory
        return (TF(fe + df, fi, fe_ext, fi_ext, W) - 2 * TF(fe, fi, fe_ext, fi_ext, W)
                + TF(fe - df, fi, fe_ext, fi_ext, W)) / ((df * 1e3) ** 2)

    model_test._diff2_fe_fe_e = _diff2_fe_fe_e

    def _diff2_fe_fe_i(fe, fi, fe_ext, fi_ext, W, df=1e-10):
        TF = model_test.TF_inhibitory
        return (TF(fe + df, fi, fe_ext, fi_ext, W) - 2 * TF(fe, fi, fe_ext, fi_ext, W)
                + TF(fe - df, fi, fe_ext, fi_ext, W)) / ((df * 1e3) ** 2)

    model_test._diff2_fe_fe_i = _diff2_fe_fe_i

    def _diff2_fe_fi(TF, fe, fi, fe_ext, fi_ext, W, df=1e-10):
        return (_diff_fe(TF, fe, fi + df, fe_ext, fi_ext, W)
                - _diff_fe(TF, fe, fi - df, fe_ext, fi_ext, W)) / (2 * df * 1e3)

    model_test._diff2_fe_fi = _diff2_fe_fi

    def _diff2_fi_fi_e(fe, fi, fe_ext, fi_ext, W, df=1e-10):
        TF = model_test.TF_excitatory
        return (TF(fe, fi + df, fe_ext, fi_ext, W) - 2 * TF(fe, fi, fe_ext, fi_ext, W)
                + TF(fe, fi - df, fe_ext, fi_ext, W)) / ((df * 1e3) ** 2)

    model_test._diff2_fi_fi_e = _diff2_fi_fi_e

    def _diff2_fi_fi_i(fe, fi, fe_ext, fi_ext, W, df=1e-10):
        TF = model_test.TF_inhibitory
        return (TF(fe, fi + df, fe_ext, fi_ext, W) - 2 * TF(fe, fi, fe_ext, fi_ext, W)
                + TF(fe, fi - df, fe_ext, fi_ext, W)) / ((df * 1e3) ** 2)

    model_test._diff2_fi_fi_i = _diff2_fi_fi_i
    return model_test


def plot_derivation(save_name, name, index, values, i_adp, adaptation_values, data, type):
    """
    plot derivation for evaluate negative one
    :param save_name: add the name for saving
    :param name: name of the measure (TF, second derivation, ...)
    :param index: index of the range explore
    :param values: values of the range of unit
    :param i_adp: index of range for adaptation
    :param adaptation_values: value of adaptation
    :param data: result of the derivation
    :param type: type of neurons
    :return:
    """
    if np.where(np.isnan(data))[0].shape[0] != 0:
        data[np.where(np.isnan(data))] = 0.0
        print(name, values[np.where(np.isnan(data))[0]], adaptation_values[np.where(np.isnan(data))[0]])
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(projection='3d')
    X, Y = np.meshgrid(adaptation_values, values)
    Z = data[:, :, 0]
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, rcount=1000,
                           vmin=-np.max(np.abs((np.nanmax(Z), np.nanmin(Z)))),
                           vmax=np.max(np.abs((np.nanmax(Z), np.nanmin(Z)))))
    fig.colorbar(surf, shrink=0.5, aspect=10)
    plt.title(name)
    plt.savefig(save_name + type + "_" + str(index) + "_" + str(i_adp) + "_" + name)
    if np.nanmin(np.abs(Z)) < 0:
        fig = plt.figure(figsize=(20, 20))
        ax = fig.add_subplot(projection='3d')
        X, Y = np.meshgrid(adaptation_values, values)
        Z = data[:, :, 0]
        surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, rcount=1000,
                               vmin=np.nanmin(Z), vmax=0.0)
        ax.set_zlim(zmax=0.1)
        surf.cmap.set_over('w')
        fig.colorbar(surf, shrink=0.5, aspect=10)
        plt.title(name + ' negative')
        plt.savefig(save_name + "neg_" + type + "_" + str(index) + "_" + str(i_adp) + "_" + name)
    else:
        print("no negative values  for : ", name)
    plt.close('all')


def plot_derivation_check(save_name, index, values, i_adp, adaptation_values, TF, diff2_fe_fe, diff2_fi_fi, diff2_fe_fi,
                          max_cut, min_scale, subtitle, type, ylabel, cmap=cm.get_cmap('viridis')):
    """

    :param save_name: add the name for saving
    :param index: index of the range explore
    :param values: values of the range of unit
    :param i_adp: index of range for adaptation
    :param adaptation_values: value of adaptation
    :param TF: Transfer Function
    :param diff2_fe_fe: second partial derivation on fe
    :param diff2_fi_fi: second partial derivation on fi
    :param diff2_fe_fi: second partial derivation on fe fi
    :param max_cut: maximum values for plotting
    :param min_scale: minimum of scale
    :param subtitle: subtitle of the figure
    :param type:  type of population
    :param ylabel: label of explored input
    :param cmap: color map
    :return:
    """
    fig, axs = plt.subplots(2, 2, figsize=(20, 20))
    fig.suptitle(subtitle)

    max_limit = np.nanmax(TF) if np.nanmax(TF) < max_cut else max_cut
    min_limit = np.nanmin(TF) if -np.nanmin(TF) < max_cut else -max_cut
    if np.nanmin(TF) >= 0.0:
        im_0 = axs[0, 0].imshow(TF, cmap=cmap, norm=Normalize(vmin=min_limit, vmax=max_limit))
    else:
        linthresh = np.min([np.nanmax(TF), -np.nanmin(TF)]) * min_scale
        im_0 = axs[0, 0].imshow(TF, cmap=cmap,
                                norm=SymLogNorm(linthresh=linthresh, vmin=min_limit, vmax=max_limit))
    fig.colorbar(im_0, ax=axs[0, 0])
    ticks_positions = np.array(np.around(np.linspace(0, values.shape[0] - 1, 7)), dtype=int)
    axs[0, 0].set_yticks(ticks_positions)
    axs[0, 0].set_yticklabels(np.around(values[ticks_positions], decimals=3) * 1e3)
    axs[0, 0].set_ylabel(ylabel)
    ticks_positions = np.array(np.around(np.linspace(0, adaptation_values.shape[0] - 1, 7)), dtype=int)
    axs[0, 0].set_xticks(ticks_positions)
    axs[0, 0].set_xticklabels(np.around(adaptation_values[ticks_positions], decimals=0))
    axs[0, 0].set_xlabel("adaptation")
    axs[0, 0].set_title('Transfer function of '+type+' neurons in Hz')
    if np.nanmin(TF) < 0.0:
        axs[0, 0].contourf(np.arange(0, TF.shape[0], 1), np.arange(0, TF.shape[1], 1),
                           TF[:, :, 0], levels=[-10000.0, 0.0], hatches=['/'], colors=['red'], alpha=0.0)
        axs[0, 0].contour(np.arange(0, TF.shape[0], 1), np.arange(0, TF.shape[1], 1),
                          TF[:, :, 0], levels=[0.0], colors=['black'], linewidths=0.1)

    max_limit = np.nanmax(diff2_fe_fe) if np.nanmax(diff2_fe_fe) < max_cut else max_cut
    min_limit = np.nanmin(diff2_fe_fe) if -np.nanmin(diff2_fe_fe) < max_cut else -max_cut
    if np.nanmin(diff2_fe_fe) > 0.0:
        linthresh = np.nanmin(diff2_fe_fe)
    elif np.nanmin(diff2_fe_fe) != 0.0:
        linthresh = np.min([np.nanmax(diff2_fe_fe), -np.nanmin(diff2_fe_fe)])*min_scale
    else:
        linthresh = np.min([np.nanmax(diff2_fe_fe)*min_scale, min_scale])
    im_1 = axs[0, 1].imshow(diff2_fe_fe, cmap=cmap,
                            norm=SymLogNorm(linthresh=linthresh, vmin=min_limit, vmax=max_limit))
    fig.colorbar(im_1, ax=axs[0, 1])
    axs[0, 1].set_title('$\partial^2f_e/\partial^2f$ TF')
    ticks_positions = np.array(np.around(np.linspace(0, values.shape[0] - 1, 7)), dtype=int)
    axs[0, 1].set_yticks(ticks_positions)
    axs[0, 1].set_yticklabels(np.around(values[ticks_positions], decimals=3) * 1e3)
    axs[0, 1].set_ylabel(ylabel)
    ticks_positions = np.array(np.around(np.linspace(0, adaptation_values.shape[0] - 1, 7)), dtype=int)
    axs[0, 1].set_xticks(ticks_positions)
    axs[0, 1].set_xticklabels(np.around(adaptation_values[ticks_positions], decimals=0))
    axs[0, 1].set_xlabel("adaptation")
    if np.nanmin(diff2_fe_fe) < 0.0:
        axs[0, 1].contourf(np.arange(0, diff2_fe_fe.shape[1], 1), np.arange(0, diff2_fe_fe.shape[0], 1),
                           diff2_fe_fe[:, :, 0], levels=[-10000.0, 0.0], hatches=['/'], colors=['red'], alpha=0.0)
        axs[0, 1].contour(np.arange(0, diff2_fe_fe.shape[1], 1), np.arange(0, diff2_fe_fe.shape[0], 1),
                          diff2_fe_fe[:, :, 0], levels=[0.0], colors=['black'], linewidths=0.1)

    max_limit = np.nanmax(diff2_fi_fi) if np.nanmax(diff2_fi_fi) < max_cut else max_cut
    min_limit = np.nanmin(diff2_fi_fi) if -np.nanmin(diff2_fi_fi) < max_cut else -max_cut
    if np.nanmin(diff2_fi_fi) > 0.0:
        linthresh = np.nanmin(diff2_fi_fi)
    elif np.nanmin(diff2_fi_fi) != 0.0:
        linthresh = np.min([np.nanmax(diff2_fi_fi), -np.nanmin(diff2_fi_fi)])*min_scale
    else:
        linthresh = np.min([np.nanmax(diff2_fi_fi)*min_scale, min_scale])
    im_2 = axs[1, 0].imshow(diff2_fi_fi, cmap=cmap,
                            norm=SymLogNorm(linthresh=linthresh, vmin=min_limit, vmax=max_limit))
    fig.colorbar(im_2, ax=axs[1, 0])
    axs[1, 0].set_title('$\partial^2f_i/\partial^2f$ TF')
    ticks_positions = np.array(np.around(np.linspace(0, values.shape[0] - 1, 7)), dtype=int)
    axs[1, 0].set_yticks(ticks_positions)
    axs[1, 0].set_yticklabels(np.around(values[ticks_positions], decimals=3) * 1e3)
    axs[1, 0].set_ylabel(ylabel)
    ticks_positions = np.array(np.around(np.linspace(0, adaptation_values.shape[0] - 1, 7)), dtype=int)
    axs[1, 0].set_xticks(ticks_positions)
    axs[1, 0].set_xticklabels(np.around(adaptation_values[ticks_positions], decimals=0))
    axs[1, 0].set_xlabel("adaptation")
    if np.nanmin(diff2_fi_fi) < 0.0:
        axs[1, 0].contourf(np.arange(0, diff2_fi_fi.shape[1], 1), np.arange(0, diff2_fi_fi.shape[0], 1),
                           diff2_fi_fi[:, :, 0], levels=[-10000.0, 0.0], hatches=['/'], colors=['red'], alpha=0.0)
        axs[1, 0].contour(np.arange(0, diff2_fi_fi.shape[1], 1), np.arange(0, diff2_fi_fi.shape[0], 1),
                          diff2_fi_fi[:, :, 0], levels=[0.0], colors=['black'], linewidths=0.1)

    max_limit = np.nanmax(diff2_fe_fi) if np.nanmax(diff2_fe_fi) < max_cut else max_cut
    min_limit = np.nanmin(diff2_fe_fi) if -np.nanmin(diff2_fe_fi) < max_cut else -max_cut
    if np.nanmin(diff2_fe_fi) > 0.0:
        linthresh = np.nanmin(diff2_fe_fi)
    elif np.nanmin(diff2_fe_fi) != 0.0:
        linthresh = np.min([np.nanmax(diff2_fe_fi), -np.nanmin(diff2_fe_fi)])*min_scale
    else:
        linthresh = np.min([np.nanmax(diff2_fe_fi)*min_scale, min_scale])
    im_3 = axs[1, 1].imshow(diff2_fe_fi, cmap=cmap,
                            norm=SymLogNorm(linthresh=linthresh, vmin=min_limit, vmax=max_limit))
    fig.colorbar(im_3, ax=axs[1, 1])
    ticks_positions = np.array(np.around(np.linspace(0, values.shape[0] - 1, 7)), dtype=int)
    axs[1, 1].set_yticks(ticks_positions)
    axs[1, 1].set_yticklabels(np.around(values[ticks_positions], decimals=3) * 1e3)
    axs[1, 1].set_ylabel(ylabel)
    ticks_positions = np.array(np.around(np.linspace(0, adaptation_values.shape[0] - 1, 7)), dtype=int)
    axs[1, 1].set_xticks(ticks_positions)
    axs[1, 1].set_xticklabels(np.around(adaptation_values[ticks_positions], decimals=0))
    axs[1, 1].set_xlabel("adaptation")
    axs[1, 1].set_title('$\partial f_i\partial f_e/\partial^2f$ TF')
    if np.nanmin(diff2_fe_fi) < 0.0:
        axs[1, 1].contourf(np.arange(0, diff2_fe_fi.shape[1], 1), np.arange(0, diff2_fe_fi.shape[0], 1),
                           diff2_fe_fi[:, :, 0], levels=[-10000.0, 0.0], hatches=['/'], colors=['red'], alpha=0.0)
        axs[1, 1].contour(np.arange(0, diff2_fe_fi.shape[1], 1), np.arange(0, diff2_fe_fi.shape[0], 1),
                          diff2_fe_fi[:, :, 0], levels=[0.0], colors=['black'], linewidths=0.1)
    plt.savefig(save_name + type+"_" + str(index) + "_" + str(i_adp) + '.png')

    plt.close('all')


def plot_derivation_check_neg(save_name, index, values, i_adp, adaptation_values, TF, diff2_fe_fe, diff2_fi_fi,
                              diff2_fe_fi, max_cut, min_scale, subtitle, type, ylabel, cmap=cm.get_cmap('viridis')):
    """

    :param save_name: add the name for saving
    :param index: index of the range explore
    :param values: values of the range of unit
    :param i_adp: index of range for adaptation
    :param adaptation_values: value of adaptation
    :param TF: Transfer Function
    :param diff2_fe_fe: second partial derivation on fe
    :param diff2_fi_fi: second partial derivation on fi
    :param diff2_fe_fi: second partial derivation on fe fi
    :param max_cut: maximum values for plotting
    :param min_scale: minimum of scale
    :param subtitle: subtitle of the figure
    :param type:  type of population
    :param ylabel: label of explored input
    :param cmap: color map
    :return:
    """
    fig, axs = plt.subplots(2, 2, figsize=(20, 20))
    fig.suptitle(subtitle)

    max_limit = np.nanmax(TF) if np.nanmax(TF) < max_cut else max_cut
    min_limit = np.nanmin(TF) if -np.nanmin(TF) < max_cut else -max_cut
    if np.nanmin(TF) >= 0.0:
        im_0 = axs[0, 0].imshow(TF, cmap=cmap, norm=Normalize(vmin=min_limit, vmax=max_limit))
    else:
        im_0 = axs[0, 0].imshow(TF, cmap=cmap, norm=Normalize(vmin=min_limit, vmax=0.0))
    fig.colorbar(im_0, ax=axs[0, 0])
    ticks_positions = np.array(np.around(np.linspace(0, values.shape[0] - 1, 7)), dtype=int)
    axs[0, 0].set_yticks(ticks_positions)
    axs[0, 0].set_yticklabels(np.around(values[ticks_positions], decimals=3) * 1e3)
    axs[0, 0].set_ylabel(ylabel)
    ticks_positions = np.array(np.around(np.linspace(0, adaptation_values.shape[0] - 1, 7)), dtype=int)
    axs[0, 0].set_xticks(ticks_positions)
    axs[0, 0].set_xticklabels(np.around(adaptation_values[ticks_positions], decimals=0))
    axs[0, 0].set_xlabel("adaptation")
    axs[0, 0].set_title('Transfer function of '+type+' neurons in Hz')

    max_limit = np.nanmax(diff2_fe_fe) if np.nanmax(diff2_fe_fe) < max_cut else max_cut
    min_limit = np.nanmin(diff2_fe_fe) if -np.nanmin(diff2_fe_fe) < max_cut else -max_cut
    im_1 = axs[0, 1].imshow(diff2_fe_fe, cmap=cmap, norm=Normalize(vmin=min_limit, vmax=0.0))
    fig.colorbar(im_1, ax=axs[0, 1])
    axs[0, 1].set_title('$\partial^2f_e/\partial^2f$ TF')
    ticks_positions = np.array(np.around(np.linspace(0, values.shape[0] - 1, 7)), dtype=int)
    axs[0, 1].set_yticks(ticks_positions)
    axs[0, 1].set_yticklabels(np.around(values[ticks_positions], decimals=3) * 1e3)
    axs[0, 1].set_ylabel(ylabel)
    ticks_positions = np.array(np.around(np.linspace(0, adaptation_values.shape[0] - 1, 7)), dtype=int)
    axs[0, 1].set_xticks(ticks_positions)
    axs[0, 1].set_xticklabels(np.around(adaptation_values[ticks_positions], decimals=0))
    axs[0, 1].set_xlabel("adaptation")

    max_limit = np.nanmax(diff2_fi_fi) if np.nanmax(diff2_fi_fi) < max_cut else max_cut
    min_limit = np.nanmin(diff2_fi_fi) if -np.nanmin(diff2_fi_fi) < max_cut else -max_cut
    im_2 = axs[1, 0].imshow(diff2_fi_fi, cmap=cmap, norm=Normalize(vmin=min_limit, vmax=0.0))
    fig.colorbar(im_2, ax=axs[1, 0])
    axs[1, 0].set_title('$\partial^2f_i/\partial^2f$ TF')
    ticks_positions = np.array(np.around(np.linspace(0, values.shape[0] - 1, 7)), dtype=int)
    axs[1, 0].set_yticks(ticks_positions)
    axs[1, 0].set_yticklabels(np.around(values[ticks_positions], decimals=3) * 1e3)
    axs[1, 0].set_ylabel(ylabel)
    ticks_positions = np.array(np.around(np.linspace(0, adaptation_values.shape[0] - 1, 7)), dtype=int)
    axs[1, 0].set_xticks(ticks_positions)
    axs[1, 0].set_xticklabels(np.around(adaptation_values[ticks_positions], decimals=0))
    axs[1, 0].set_xlabel("adaptation")

    max_limit = np.nanmax(diff2_fe_fi) if np.nanmax(diff2_fe_fi) < max_cut else max_cut
    min_limit = np.nanmin(diff2_fe_fi) if -np.nanmin(diff2_fe_fi) < max_cut else -max_cut
    im_3 = axs[1, 1].imshow(diff2_fe_fi, cmap=cmap, norm=Normalize(vmin=min_limit, vmax=0.0))
    fig.colorbar(im_3, ax=axs[1, 1])
    ticks_positions = np.array(np.around(np.linspace(0, values.shape[0] - 1, 7)), dtype=int)
    axs[1, 1].set_yticks(ticks_positions)
    axs[1, 1].set_yticklabels(np.around(values[ticks_positions], decimals=3) * 1e3)
    axs[1, 1].set_ylabel(ylabel)
    ticks_positions = np.array(np.around(np.linspace(0, adaptation_values.shape[0] - 1, 7)), dtype=int)
    axs[1, 1].set_xticks(ticks_positions)
    axs[1, 1].set_xticklabels(np.around(adaptation_values[ticks_positions], decimals=0))
    axs[1, 1].set_xlabel("adaptation")
    axs[1, 1].set_title('$\partial f_i\partial f_e/\partial^2f$ TF')
    plt.savefig('neg_' + save_name + type+"_" + str(index) + "_" + str(i_adp) + '.png')

    plt.close('all')