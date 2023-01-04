#  Copyright 2021 Aix-Marseille Université
# "Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements; and to You under the Apache License, Version 2.0. "
import matplotlib.pyplot as plt
from matplotlib.text import Text
import matplotlib as mpl
import matplotlib.cm as cm
import numpy as np


def plot_matrix(result_n):
    """
    plot result in form of matrix
    :param result_n: result of the fitting
    :return:
    """
    # plot degree  of firing rate ( not very interesting )
    for i in range(result_n.shape[2]):
        fig, axs = plt.subplots(2, 2, figsize=(20, 20))
        axis = axs[0, 0]
        mat = axis.imshow(result_n[:, :, i, 0])
        axis.set_title('F_out')
        fig.colorbar(mat, ax=axis, extend='both')
        axis = axs[0, 1]
        mat = axis.imshow(result_n[:, :, i, 1])
        axis.set_title('Fe_in')
        fig.colorbar(mat, ax=axis, extend='both')
        axis = axs[1, 0]
        mat = axis.imshow(result_n[:, :, i, 2])
        axis.set_title('Fi_in')
        fig.colorbar(mat, ax=axis, extend='both')
        axis = axs[1, 1]
        mat = axis.imshow(result_n[:, :, i, 3])
        fig.colorbar(mat, ax=axis, extend='both')
        axis.set_title('adaptation')
        fig.suptitle('adaptation = ' + str(result_n[:, :, i, 3][0][0]), fontsize=16)


def plot_result_std(result_n, name_fig):
    """
    plot data with standard deviation of the result
    :param result_n: data
    :param name_fig: name of figure for saving
    :return:
    """
    for i in range(result_n.shape[2]):
        fig = plt.figure(figsize=(20, 20))
        # print mean
        plt.plot(result_n[:, :, i, 1] * 1e3, result_n[:, :, i, 0] * 1e3, 'g')
        # print mean + std
        plt.plot(result_n[:, :, i, 1] * 1e3, result_n[:, :, i, 4] * 1e3 + result_n[:, :, i, 0] * 1e3, 'r')
        # print mean - std
        plt.plot(result_n[:, :, i, 1] * 1e3, -result_n[:, :, i, 4] * 1e3 + result_n[:, :, i, 0] * 1e3, 'b')
        plt.title('adapt = ' + str(result_n[:, :, i, 3][0][0]), {"fontsize": 30.0})
        plt.ylabel('output frequency of the neurons in Hz', {"fontsize": 30.0})
        plt.xlabel('excitatory input frequency in Hz', {"fontsize": 30.0})
        plt.tick_params(labelsize=10.0)
        np.set_printoptions(precision=2)
        fig.add_artist(Text(0.5, 0.01, "frequency inhibitory " + str(repr(result_n[:, :, i, 2][0] * 1e3)), ha='center',
                            fontsize=20.0))
        plt.subplots_adjust(bottom=0.18)
        name_fig_i = name_fig + str(i) + '.svg'
        plt.savefig(name_fig_i)

        # print coefficient of variation
        fig = plt.figure(figsize=(20, 20))
        plt.plot(result_n[:, :, i, 1] * 1e3, result_n[:, :, i, 4] / result_n[:, :, i, 0])
        name_fig_i = 'std_' + str(i) + '.svg'
        plt.savefig(name_fig_i)
        plt.close('all')


def plot_result_box_plot(result_n, name_fig, nb_value_finh, nb_value_fexc):
    """
    plot the result in format of box plot of the data
    :param result_n: data
    :param name_fig: name of the figure
    :param nb_value_finh: number of inhibitory input frequency
    :param nb_value_fexc: number of inhibitory input frequency
    :return:
    """
    norm = mpl.colors.Normalize(vmin=0, vmax=nb_value_fexc * nb_value_finh)
    cmap = cm.prism
    m = cm.ScalarMappable(norm=norm, cmap=cmap)

    for i in range(result_n.shape[2]):
        fig = plt.figure(figsize=(20, 20))
        for j in range(nb_value_finh):
            for k in range(nb_value_fexc):
                color_line = m.to_rgba(k + j * nb_value_fexc)
                whiskerprops = {'color': color_line}
                meanpointprops = dict(marker='D', markeredgecolor='orange', markerfacecolor='red', markersize=0.5)
                medianprops = dict(color='red', linewidth=1.0)
                point = dict(markerfacecolor=color_line, marker='.', linestyle='none', markeredgewidth=0.1,
                             markersize=5.0)
                plt.boxplot(result_n[k, j, i, 6:] * 1e3, positions=[result_n[k, j, i, 1] * 1e3], notch=True,
                            widths=0.00001, flierprops=point,
                            showmeans=True, showbox=False, whiskerprops=whiskerprops, meanprops=meanpointprops,
                            medianprops=medianprops
                            )
        plt.ylim(ymax=np.max(result_n[:, :, i, 6:] * 1e3 + 0.5))
        plt.title('adapt = ' + str(result_n[:, :, i, 3][0][0]), {"fontsize": 30.0})
        plt.ylabel('output frequency of the neurons in Hz', {"fontsize": 30.0})
        plt.xlabel('excitatory input frequency in Hz', {"fontsize": 30.0})
        plt.tick_params(labelsize=10.0)
        np.set_printoptions(precision=2)
        fig.add_artist(Text(0.5, 0.01, "frequency inhibitory " + str(repr(result_n[:, :, i, 2][0] * 1e3)), ha='center',
                            fontsize=20.0))
        plt.subplots_adjust(bottom=0.1, top=0.9)
        name_fig_i = name_fig + 'box_plot_' + str(i) + '.svg'
        plt.savefig(name_fig_i, dpi=600)
        plt.close('all')


def plot_result_curve_box_std(result_n, name_fig, nb_value_finh, nb_value_fexc):
    """
    plot result with curve, box plot and standard deviation
    :param result_n: data
    :param name_fig: name of the figure
    :param nb_value_finh: number of inhibitory input frequency
    :param nb_value_fexc: number of excitatory input frequency
    :return:
    """
    norm = mpl.colors.Normalize(vmin=0, vmax=nb_value_fexc)
    cmap = cm.prism
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    np.set_printoptions(precision=2)

    for i in range(result_n.shape[2]):
        for j in range(nb_value_finh):
            fig = plt.figure(figsize=(20, 20))
            for k in range(nb_value_fexc):
                color_line = m.to_rgba(k)
                whiskerprops = {'color': color_line}
                meanpointprops = dict(marker='D', markeredgecolor='orange', markerfacecolor='red', markersize=0.5)
                medianprops = dict(color='red', linewidth=1.0)
                point = dict(markerfacecolor=color_line, marker='.', linestyle='none', markeredgewidth=0.1,
                             markersize=5.0)
                plt.boxplot(result_n[k, j, i, 6:] * 1e3, positions=[result_n[k, j, i, 1] * 1e3], notch=True,
                            widths=0.00001, flierprops=point,
                            showmeans=True, showbox=False, whiskerprops=whiskerprops, meanprops=meanpointprops,
                            medianprops=medianprops
                            )
            plt.xticks(result_n[:, j, i, 1] * 1e3, np.around(result_n[:, j, i, 1] * 1e3, 2))
            # print mean
            plt.plot(result_n[:, j, i, 1] * 1e3, result_n[:, j, i, 0] * 1e3, 'g', alpha=0.5)
            # print median
            plt.plot(result_n[:, j, i, 1] * 1e3, result_n[:, j, i, 5] * 1e3, 'orange', alpha=0.5)
            # print mean + std
            plt.plot(result_n[:, j, i, 1] * 1e3, result_n[:, j, i, 4] * 1e3 + result_n[:, j, i, 0] * 1e3, 'r',
                     alpha=0.5)
            # print mean - std
            plt.plot(result_n[:, j, i, 1] * 1e3, -result_n[:, j, i, 4] * 1e3 + result_n[:, j, i, 0] * 1e3, 'b',
                     alpha=0.5)
            plt.ylim(ymax=np.max(result_n[:, j, i, 6:] * 1e3 + 0.5))
            plt.title('adapt = ' + str(result_n[:, j, i, 3][0]) + ' inhibition :' + str(result_n[:, j, i, 2][0] * 1e3),
                      {"fontsize": 30.0})
            plt.ylabel('output frequency of the neurons in Hz', {"fontsize": 30.0})
            plt.xlabel('excitatory input frequency in Hz', {"fontsize": 30.0})
            plt.tick_params(labelsize=2.0)
            name_fig_i = name_fig + 'curve_adp_' + str(i) + '_inh_' + str(j) + '.svg'
            plt.savefig(name_fig_i, dpi=600)
            plt.close('all')


def plot_result(result_n, function_fit, P_relatif, P_absolute, name_fig, nb_value_finh, nb_value_fexc):
    """
    plot result and compare with mean field
    :param result_n: data
    :param function_fit: fitting function
    :param P_relatif: polynomial of first fitting
    :param P_absolute: polynomial of second fitting
    :param name_fig: name of figure
    :param nb_value_finh: number of inhibitory input frequency
    :param nb_value_fexc: number of excitatory input frequency
    :return:
    """
    for i in range(result_n.shape[2]):
        fig = plt.figure(figsize=(20, 20))
        # # real data
        # plt.plot(result_n[:,:,i,1]*1e3,result_n[:,:,i,0]*1e3, '.g')
        # # fit data
        # line_1 = plt.plot(result_n[:,:,i,1]*1e3,function_fit(P_relatif,result_n[:,:,i,1],result_n[:,:,i,2] , result_n[:,:,i,3])*1e3,'c')
        # line_2 = plt.plot(result_n[:,:,i,1]*1e3,function_fit(P_absolute,result_n[:,:,i,1],result_n[:,:,i,2] , result_n[:,:,i,3])*1e3,'m')
        # if i == 0 :
        #     line_1[0].set_label('relatif')
        #     line_2[0].set_label('absolute')
        plt.title('adapt = ' + str(result_n[:, :, i, 3][0][0]), {"fontsize": 30.0})
        plt.ylabel('output frequency of the neurons in Hz', {"fontsize": 30.0})
        plt.xlabel('excitatory input frequency in Hz', {"fontsize": 30.0})
        plt.tick_params(labelsize=10.0)
        np.set_printoptions(precision=2)
        plt.legend()
        fig.add_artist(Text(0.5, 0.01, "frequency inhibitory " + str(repr(result_n[:, :, i, 2][0] * 1e3)), ha='center',
                            fontsize=20.0))
        plt.subplots_adjust(bottom=0.18)
        for j in range(nb_value_finh):
            for k in range(nb_value_fexc):
                if not np.isnan(result_n[k, j, i, 0]):
                    plt.plot([result_n[k, j, i, 1] * 1e3, result_n[k, j, i, 1] * 1e3],
                             [result_n[k, j, i, 0] * 1e3,
                              function_fit(P_relatif, result_n[k, j, i, 1], result_n[k, j, i, 2],
                                           result_n[k, j, i, 3]) * 1e3], color='r', alpha=0.5, linewidth=0.5,
                             linestyle='dashed', )
                    plt.plot([result_n[k, j, i, 1] * 1e3, result_n[k, j, i, 1] * 1e3],
                             [result_n[k, j, i, 0] * 1e3,
                              function_fit(P_absolute, result_n[k, j, i, 1], result_n[k, j, i, 2],
                                           result_n[k, j, i, 3]) * 1e3], color='k', alpha=0.5, linewidth=0.5,
                             linestyle='dashed', )
        plt.ylim(ymax=50.0)
        name_fig_i = name_fig + str(i) + '.svg'
        plt.savefig(name_fig_i)
        plt.close('all')


def print_result_1(result_n, function_fit, P, name_fig, nb_value_finh, nb_value_fexc):
    """
    plot result and compare with mean field
    :param result_n: data
    :param function_fit: fitting function
    :param P: polynomial
    :param name_fig: name of figure
    :param nb_value_finh: number of inhibitory input frequency
    :param nb_value_fexc: number of excitatory input frequency
    :return:
    """
    for i in range(result_n.shape[2]):
        fig = plt.figure(figsize=(20, 20))
        # # real data
        # plt.plot(result_n[:,:,i,1]*1e3,result_n[:,:,i,0]*1e3, '.g')
        # # fit data
        # line = plt.plot(result_n[:,:,i,1]*1e3,function_fit(P_absolute,result_n[:,:,i,1],result_n[:,:,i,2] , result_n[:,:,i,3])*1e3,'m')
        plt.title('adapt = ' + str(result_n[:, :, i, 3][0][0]), {"fontsize": 30.0})
        plt.ylabel('output frequency of the neurons in Hz', {"fontsize": 30.0})
        plt.xlabel('excitatory input frequency in Hz', {"fontsize": 30.0})
        plt.tick_params(labelsize=10.0)
        np.set_printoptions(precision=2)
        plt.legend()
        fig.add_artist(Text(0.5, 0.01, "frequency inhibitory " + str(repr(result_n[:, :, i, 2][0] * 1e3)), ha='center',
                            fontsize=20.0))
        plt.subplots_adjust(bottom=0.18)
        for j in range(nb_value_finh):
            for k in range(nb_value_fexc):
                if not np.isnan(result_n[k, j, i, 0]):
                    plt.plot([result_n[k, j, i, 1] * 1e3, result_n[k, j, i, 1] * 1e3],
                             [result_n[k, j, i, 0] * 1e3,
                              function_fit(P, result_n[k, j, i, 1], result_n[k, j, i, 2], result_n[k, j, i, 3]) * 1e3],
                             color='r', alpha=0.5, linewidth=0.5)
        plt.ylim(ymax=50.0)
        name_fig_i = name_fig + str(i) + '.svg'
        plt.savefig(name_fig_i)
        plt.close('all')


def plot_result_zerlaut(result_n, TF, p_with, p_without, name_fig, nb_value_finh, nb_value_fexc):
    """
    plot error between mean field and data for each inhibitory input and adaptation
    :param result_n: data
    :param TF: transfer function
    :param p_with: polynomial fitting with adaptation
    :param p_without: polynomial fitting without adaptation
    :param name_fig: name of figure for saving
    :param nb_value_finh: number of inhibitory input frequency
    :param nb_value_fexc: number of excitatory input frequency
    :return:
    """
    for curve, adaptation, real_data in [
        (False, False, False), (True, False, False), (False, False, True),(True, False, True),
        (False, True, False), (True, True, False), (False, True, True), (True, True, True)]:
        print(curve, adaptation, real_data)
        # plot zerlaut fitting and the error
        for i in [0, result_n.shape[2]-1]:
            # print error without adaptation
            for j in range(nb_value_finh):
                fig = plt.figure(figsize=(20, 20))
                # # real data
                if real_data:
                    plt.plot(result_n[:, j, i, 1] * 1e3, result_n[:, j, i, 0] * 1e3, '.g')
                if curve:
                    if adaptation:
                        # fit with adaptation
                        plt.plot(result_n[:, j, i, 1] * 1e3,
                                 TF(result_n[:, j, i, 1], result_n[:, j, i, 2], p_with, w=result_n[:, j, i, 3]) * 1e3, 'c')
                    else:
                        # fit without adaptation
                        plt.plot(result_n[:, j, i, 1] * 1e3,
                                 TF(result_n[:, j, i, 1], result_n[:, j, i, 2], p_without, w=result_n[:, j, i, 3]) * 1e3,
                                 '--b')
                plt.title('adapt = ' + str(result_n[:, j, i, 3][0]), {"fontsize": 30.0})
                plt.ylabel('output frequency of the neurons in Hz', {"fontsize": 30.0})
                plt.xlabel('excitatory input frequency in Hz', {"fontsize": 30.0})
                plt.tick_params(labelsize=10.0)
                np.set_printoptions(precision=2)
                fig.add_artist(
                    Text(0.5, 0.01, "frequency inhibitory " + str(repr(result_n[:, j, i, 2][0] * 1e3)), ha='center',
                         fontsize=20.0))
                plt.subplots_adjust(bottom=0.18)
                for k in range(nb_value_fexc):
                    if not np.isnan(result_n[k, j, i, 0]):
                        if adaptation:
                            plt.plot([result_n[k, j, i, 1] * 1e3, result_n[k, j, i, 1] * 1e3],
                                     [result_n[k, j, i, 0] * 1e3, TF(result_n[k, j, i, 1], result_n[k, j, i, 2],
                                                                     p_with, w=result_n[k, j, i, 3]) * 1e3],
                                     color='r', alpha=0.5, linewidth=0.5)
                        else:
                            # print error with adaptation
                            plt.plot([result_n[k, j, i, 1] * 1e3, result_n[k, j, i, 1] * 1e3],
                                     [result_n[k, j, i, 0] * 1e3, TF(result_n[k, j, i, 1], result_n[k, j, i, 2],
                                                                     p_without, w=result_n[k, j, i, 3]) * 1e3], color='r',
                                     alpha=0.5, linewidth=0.5)
                name_fig_i = name_fig
                if real_data:
                    name_fig_i += '_data_'
                if curve:
                    name_fig_i += '_curve_'
                if adaptation:
                    name_fig_i += '_adaptation_'
                name_fig_i += str(i) + '_' + str(j) + '.svg'
                plt.ylim(ymax=200.0, ymin=0.0)
                plt.xlim(xmin=0.0)
                # plt.show()
                plt.savefig(name_fig_i)
                plt.close('all')

def plot_result_zerlaut_all(result_n, TF, p_with, p_without, name_fig, nb_value_finh, nb_value_fexc):
    """
    plot error between mean field and data for each adaptation
    :param result_n: data
    :param TF: transfer function
    :param p_with: polynomial fitting with adaptation
    :param p_without: polynomial fitting without adaptation
    :param name_fig: name of figure for saving
    :param nb_value_finh: number of inhibitory input frequency
    :param nb_value_fexc: number of excitatory input frequency
    :return:
    """
    for curve, adaptation, real_data in [
        (False, False, False), (True, False, False), (False, False, True),(True, False, True),
        (False, True, False), (True, True, False), (False, True, True), (True, True, True)]:
        print(curve, adaptation, real_data)
        # plot zerlaut fitting and the error
        for i in [0, result_n.shape[2]-1]:
            # print error without adaptation
            fig = plt.figure(figsize=(20, 20))
            # # real data
            if real_data:
                plt.plot(result_n[:, :, i, 1] * 1e3, result_n[:, :, i, 0] * 1e3, '.g')
            if curve:
                if adaptation:
                    # fit with adaptation
                    plt.plot(result_n[:, :, i, 1] * 1e3,
                             TF(result_n[:, :, i, 1], result_n[:, :, i, 2], p_with, w=result_n[:, :, i, 3]) * 1e3, 'c')
                else:
                    # fit without adaptation
                    plt.plot(result_n[:, :, i, 1] * 1e3,
                             TF(result_n[:, :, i, 1], result_n[:, :, i, 2], p_without, w=result_n[:, :, i, 3]) * 1e3,
                             '--b')
            plt.title('adapt = ' + str(result_n[:, :, i, 3][0][0]), {"fontsize": 30.0})
            plt.ylabel('output frequency of the neurons in Hz', {"fontsize": 30.0})
            plt.xlabel('excitatory input frequency in Hz', {"fontsize": 30.0})
            plt.tick_params(labelsize=10.0)
            np.set_printoptions(precision=2)
            fig.add_artist(
                Text(0.5, 0.01, "frequency inhibitory " + str(repr(result_n[:, :, i, 2][0] * 1e3)), ha='center',
                     fontsize=20.0))
            plt.subplots_adjust(bottom=0.18)
            for j in range(nb_value_finh):
                for k in range(nb_value_fexc):
                    if not np.isnan(result_n[k, j, i, 0]):
                        if adaptation:
                            plt.plot([result_n[k, j, i, 1] * 1e3, result_n[k, j, i, 1] * 1e3],
                                     [result_n[k, j, i, 0] * 1e3, TF(result_n[k, j, i, 1], result_n[k, j, i, 2],
                                                                     p_with, w=result_n[k, j, i, 3]) * 1e3],
                                     color='r', alpha=0.5)
                        else:
                            # print error with adaptation
                            plt.plot([result_n[k, j, i, 1] * 1e3, result_n[k, j, i, 1] * 1e3],
                                     [result_n[k, j, i, 0] * 1e3, TF(result_n[k, j, i, 1], result_n[k, j, i, 2],
                                                                     p_without, w=result_n[k, j, i, 3]) * 1e3], color='r',
                                     alpha=0.5, linewidth=0.5)
            name_fig_i = name_fig
            if real_data:
                name_fig_i += '_data_'
            if curve:
                name_fig_i += '_curve_'
            if adaptation:
                name_fig_i += '_adaptation_'
            name_fig_i += str(i) + '_all.svg'
            plt.ylim(ymax=200.0, ymin=0.0)
            plt.xlim(xmin=0.0)
            # plt.show()
            plt.savefig(name_fig_i)
            plt.close('all')
