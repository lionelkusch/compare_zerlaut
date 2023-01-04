#  Copyright 2021 Aix-Marseille Université
# "Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements; and to You under the Apache License, Version 2.0. "
import os
import numpy as np
from .generate_data import generate_rates
from .fitting_function_zerlaut import fitting_model_zerlaut, create_transfer_function
from .plot.plot_result import plot_details_raw, plot_result
from .plot.plot_fitting_figure import plot_result_zerlaut
from .plot.helper_function import get_result


def engin(parameters, parameters_all, excitatory,
          MAXfexc=40., MINfexc=0., nb_value_fexc=60,
          MAXfinh=40., MINfinh=0., nb_value_finh=20,
          MAXadaptation=100., MINadaptation=0., nb_value_adaptation=20,
          MAXfout=20., MINfout=0.0, MAXJump=1.0, MINJump=0.1,
          nb_neurons=50, fitting=True,
          name_file_fig='./',
          dt=1e-4, tstop=10.0,
          print_details=False, print_fitting=True, print_error=False, print_details_raw=False):
    """
    generate polynomial of the transfer function
    :param parameters: parameters of the neurons
    :param parameters_all: parameter excitatory and inhibitory used for the model
    :param excitatory: boolean for parameter for inhibitory or excitatory
    :param MAXfexc: maximum excitatory input firing rate
    :param MINfexc: minimum firing rate
    :param nb_value_fexc: number of different value of excitatory input firing rate
    :param MAXfinh: maximum inhibitory input firing rate
    :param MINfinh: minimum inhibitory input firing rate
    :param nb_value_finh: number of different value of inhibitory input firing rate
    :param MAXadaptation: maximum of adaptation
    :param MINadaptation: minimum of adaptation
    :param nb_value_adaptation: number of different value of adaptation
    :param MAXfout: maximum of output firing rate
    :param MAXJump: maximum jump of output firing rate
    :param MINJump: minimum jump of output firing rate
    :param nb_neurons: number of trial ofr each condition
    :param name_file_fig: path for saving figure
    :param dt: step of integration
    :param tstop: time of simulation
    :param print_details: boolean for plotting element
    :param print_error: boolean for plotting error
    :param print_fitting: print result fitting
    :param print_details_raw: boolean for plotting details
    :return:
    """
    # file name
    name_file = name_file_fig
    for name, value in parameters.items():
        name_file += name + '_' + str(value) + '/'

    # check previous result :
    if os.path.exists(name_file + '/fout.npy'):
        # check if the frequency in output exit
        pass
    else:
        # generate the spikes for all the case
        generate_rates(parameters,
                       MAXfexc=MAXfexc, MINfexc=MINfexc, nb_value_fexc=nb_value_fexc,
                       MAXfinh=MAXfinh, MINfinh=MINfinh, nb_value_finh=nb_value_finh,
                       MAXadaptation=MAXadaptation, MINadaptation=MINadaptation,
                       nb_value_adaptation=nb_value_adaptation, MAXJump=MAXJump, MINJump=MINJump,
                       nb_neurons=nb_neurons, name_file=name_file, dt=dt, tstop=tstop
                       )
    # Print data
    if excitatory:
        name_file_fig += "/ex_"
    else:
        name_file_fig += "/in_"

    if print_details_raw:
        plot_details_raw(name_file, name_file_fig, MAXfout, MAXfexc, nb_value_fexc, nb_neurons,
                          MINfinh, MAXfinh, nb_value_finh, MINadaptation, MAXadaptation, nb_value_adaptation)

    result_n, data = get_result(name_file, MAXfout, MAXfexc, nb_value_fexc, nb_neurons,
               MINfinh, MAXfinh, nb_value_finh, MINadaptation, MAXadaptation, nb_value_adaptation, data_require=True)
    if print_details:
        plot_result(name_file, name_file_fig, MAXfout, MAXfexc, nb_value_fexc, nb_neurons,
                     MINfinh, MAXfinh, nb_value_finh, MINadaptation, MAXadaptation, nb_value_adaptation)
    # Fitting data Zerlaut
    p_with, p_without, TF = fitting_model_zerlaut(data[:, 0], data[:, 1], data[:, 2], data[:, 3], parameters_all,
                                                  excitatory, print_result=print_fitting, save_result=name_file,
                                                  fitting=fitting)
    if print_error:
        plot_result_zerlaut(result_n, TF, p_with, p_without, name_file_fig + 'zerlaut_', nb_value_finh, nb_value_fexc)
    return p_with