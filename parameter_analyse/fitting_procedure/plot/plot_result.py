#  Copyright 2023 Aix-Marseille Université
# "Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements; and to You under the Apache License, Version 2.0. "
from .plot_fitting_figure import plot_result_box_plot, plot_result_std, plot_result_curve_box_std, \
    plot_result_zerlaut, plot_result_zerlaut_all
from .helper_function import get_result_raw, get_result
from ..fitting_function_zerlaut import fitting_model_zerlaut
import numpy as np


def plot_details_raw(name_file, name_file_fig, MAXfout, MAXfexc, nb_value_fexc, nb_neurons,
                     MINfinh, MAXfinh, nb_value_finh, MINadaptation, MAXadaptation, nb_value_adaptation):
    """
    plot result brut
    :param name_file: folder of the result
    :param name_file_fig: folder where to save result
    :param MAXfout: maximum output firing rate
    :param MAXfexc: maximum excitatory input firing rate
    :param nb_value_fexc: number of different value of excitatory input firing rate
    :param nb_neurons: number of trial ofr each condition
    :param MAXfinh: maximum inhibitory input firing rate
    :param MINfinh: minimum inhibitory input firing rate
    :param nb_value_finh: number of different value of inhibitory input firing rate
    :param MAXadaptation: maximum of adaptation
    :param MINadaptation: minimum of adaptation
    :param nb_value_adaptation: number of different value of adaptation
    :return:
    """
    result_n_brut = get_result_raw(name_file, MAXfout, MAXfexc, nb_value_fexc, nb_neurons,
                                   MINfinh, MAXfinh, nb_value_finh, MINadaptation, MAXadaptation, nb_value_adaptation)
    plot_result_box_plot(result_n_brut, name_file_fig + '_data_brute_', nb_value_finh, nb_value_fexc)
    plot_result_std(result_n_brut, name_file_fig + '_data_brute_')
    plot_result_curve_box_std(result_n_brut, name_file_fig + '_data_brute_', nb_value_finh, nb_value_fexc)


def plot_result(name_file, name_file_fig, MAXfout, MAXfexc, nb_value_fexc, nb_neurons,
                     MINfinh, MAXfinh, nb_value_finh, MINadaptation, MAXadaptation, nb_value_adaptation):
    """
    plot result brut
    :param name_file: folder of the result
    :param name_file_fig: folder where to save result
    :param MAXfout: maximum output firing rate
    :param MAXfexc: maximum excitatory input firing rate
    :param nb_value_fexc: number of different value of excitatory input firing rate
    :param nb_neurons: number of trial ofr each condition
    :param MAXfinh: maximum inhibitory input firing rate
    :param MINfinh: minimum inhibitory input firing rate
    :param nb_value_finh: number of different value of inhibitory input firing rate
    :param MAXadaptation: maximum of adaptation
    :param MINadaptation: minimum of adaptation
    :param nb_value_adaptation: number of different value of adaptation
    :return:
    """
    result_n, data = get_result(name_file, MAXfout, MAXfexc, nb_value_fexc, nb_neurons,
                                MINfinh, MAXfinh, nb_value_finh, MINadaptation, MAXadaptation, nb_value_adaptation)
    plot_result_box_plot(result_n, name_file_fig, nb_value_finh, nb_value_fexc)
    plot_result_std(result_n, name_file_fig)
    plot_result_curve_box_std(result_n, name_file_fig, nb_value_finh, nb_value_fexc)


def plot_error(parameters, parameters_all, excitatory, name_file_fig='./',
               MAXfout=20., MAXfexc=40., nb_value_fexc=60, MAXfinh=40., MINfinh=0., nb_value_finh=20,
               MAXadaptation=100., MINadaptation=0., nb_value_adaptation=20,
               nb_neurons=50,):

    """
    plot error
    :param parameters: parameters of the neurons
    :param parameters_all: parameter excitatory and inhibitory used for the model
    :param excitatory: boolean for parameter for inhibitory or excitatory
    :param MAXfout: maximum of output firing rate
    :param MAXfexc: maximum excitatory input firing rate
    :param nb_value_fexc: number of different value of excitatory input firing rate
    :param MAXfinh: maximum inhibitory input firing rate
    :param MINfinh: minimum inhibitory input firing rate
    :param nb_value_finh: number of different value of inhibitory input firing rate
    :param MAXadaptation: maximum of adaptation
    :param MINadaptation: minimum of adaptation
    :param nb_value_adaptation: number of different value of adaptation
    :param nb_neurons: number of trial ofr each condition
    :param name_file_fig: path for saving figure
    :return:
    """
    name_file = name_file_fig
    for name, value in parameters.items():
        name_file += name + '_' + str(value) + '/'
    result_n_brut, data = get_result_raw(name_file, MAXfout, MAXfexc, nb_value_fexc, nb_neurons,
                                         MINfinh, MAXfinh, nb_value_finh, MINadaptation, MAXadaptation,
                                         nb_value_adaptation, data_require=True)
    p_with, p_without, TF = fitting_model_zerlaut(data[:, 0], data[:, 1], data[:, 2], data[:, 3], parameters_all,
                                                  excitatory, print_result=False, save_result=name_file,
                                                  fitting=False)
    plot_result_zerlaut(result_n_brut, TF, p_with, p_without, name_file_fig + 'zerlaut_', nb_value_finh, nb_value_fexc)
    plot_result_zerlaut_all(result_n_brut, TF, p_with, p_without, name_file_fig + 'zerlaut_',
                             nb_value_finh, nb_value_fexc)
