#  Copyright 2021 Aix-Marseille Université
# "Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements; and to You under the Apache License, Version 2.0. "
import logging

logging.getLogger('numba').setLevel('ERROR')
logging.getLogger('matplotlib').setLevel('ERROR')
logging.getLogger('tvb').setLevel('ERROR')
import os
import numpy as np
from .generate_data import generate_rates, remove_outlier
from .fitting_function_zerlaut import fitting_model_zerlaut, create_transfer_function
from .print_fitting_figure import print_result_box_plot, print_result_std, print_result_curve_box_std, \
    print_result_zerlaut


def engin(parameters, parameters_all, excitatory,
          MAXfexc=40., MINfexc=0., nb_value_fexc=60,
          MAXfinh=40., MINfinh=0., nb_value_finh=20,
          MAXadaptation=100., MINadaptation=0., nb_value_adaptation=20,
          MAXfout=20., MINfout=0.0, MAXJump=1.0, MINJump=0.1,
          nb_neurons=50,
          name_file_fig='./',
          dt=1e-4, tstop=10.0,
          print_details=False, print_error=False, print_details_brut=False):
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
                       nb_value_adaptation=nb_value_adaptation,
                       MAXJump=MAXJump, MINJump=MINJump,
                       nb_neurons=nb_neurons, name_file=name_file, dt=dt, tstop=tstop
                       )
    # structure data
    results = np.load(name_file + '/fout.npy').reshape(nb_value_adaptation * nb_value_fexc * nb_value_finh,
                                                       nb_neurons) * 1e-3
    # Print data
    if excitatory:
        name_file_fig += "/ex_"
    else:
        name_file_fig += "/in_"

    if print_details_brut:
        brut_feOut = np.nanmean(results, axis=1)
        brut_feOut_std = np.nanstd(results, axis=1).ravel()
        brut_feOut_med = np.median(results, axis=1).ravel()
        brut_feSim = np.load(name_file + '/fin.npy').ravel() * 1e-3
        brut_fiSim = np.repeat([np.repeat(np.linspace(MINfinh, MAXfinh, nb_value_finh), nb_value_adaptation)],
                               nb_value_fexc,
                               axis=0).ravel() * 1e-3
        brut_adaptation = np.repeat(
            [np.repeat([np.linspace(MINadaptation, MAXadaptation, nb_value_adaptation)], nb_value_finh, axis=0)],
            nb_value_fexc, axis=0).ravel()
        # shape data adn remove fe higher than some values
        i = 0
        result_n_brut = np.empty((nb_value_fexc, nb_value_finh, nb_value_adaptation, 6 + nb_neurons))
        result_n_brut[:] = np.NAN
        fe_model = -1
        np.linspace(MINadaptation, MAXadaptation, nb_value_adaptation)
        while i != len(brut_fiSim):
            fi_model = np.where(brut_fiSim[i] == np.linspace(MINfinh, MAXfinh, nb_value_finh) * 1e-3)[0][0]
            w_model = np.where(brut_adaptation[i] == np.linspace(MINadaptation, MAXadaptation, nb_value_adaptation))[0][
                0]
            if brut_adaptation[i] < brut_adaptation[i - 1]:
                if brut_fiSim[i] < brut_fiSim[i - 1]:
                    fe_model += 1
            result_n_brut[fe_model, fi_model, w_model, :6] = [brut_feOut[i], brut_feSim[i], brut_fiSim[i],
                                                              brut_adaptation[i], brut_feOut_std[i], brut_feOut_med[i]]
            result_n_brut[fe_model, fi_model, w_model, 6:] = results[i]
            i += 1
        # print_result_box_plot(result_n_brut,name_file_fig+'_data_brute_',nb_value_finh,nb_value_fexc)
        # print_result_std(result_n_brut, name_file_fig + '_data_brute_')
        # print_result_curve_box_std(result_n_brut, name_file_fig+'_data_brute_', nb_value_finh, nb_value_fexc)

    # remove outlier
    feOut = np.nanmean(remove_outlier(results), axis=1)
    feOut_std = np.nanstd(remove_outlier(results), axis=1).ravel()
    feOut_med = np.median(remove_outlier(results), axis=1).ravel()
    feSim = np.load(name_file + '/fin.npy').ravel() * 1e-3
    fiSim = np.repeat([np.repeat(np.linspace(MINfinh, MAXfinh, nb_value_finh), nb_value_adaptation)], nb_value_fexc,
                      axis=0).ravel() * 1e-3
    adaptation = np.repeat(
        [np.repeat([np.linspace(MINadaptation, MAXadaptation, nb_value_adaptation)], nb_value_finh, axis=0)],
        nb_value_fexc, axis=0).ravel()

    # shape data adn remove fe higher than some values
    i = 0
    result_n = np.empty((nb_value_fexc, nb_value_finh, nb_value_adaptation, 6 + nb_neurons))
    result_n[:] = np.NAN
    fe_model = -1
    np.linspace(MINadaptation, MAXadaptation, nb_value_adaptation)
    data = []
    while i != len(fiSim):
        fi_model = np.where(fiSim[i] == np.linspace(MINfinh, MAXfinh, nb_value_finh) * 1e-3)[0][0]
        w_model = np.where(adaptation[i] == np.linspace(MINadaptation, MAXadaptation, nb_value_adaptation))[0][0]
        if adaptation[i] < adaptation[i - 1]:
            if fiSim[i] < fiSim[i - 1]:
                fe_model += 1
        result_n[fe_model, fi_model, w_model, :6] = [feOut[i], feSim[i], fiSim[i], adaptation[i], feOut_std[i],
                                                     feOut_med[i]]
        result_n[fe_model, fi_model, w_model, 6:] = results[i]
        if not (feOut[i] > MAXfout * 1e-3):  # coefficient of variation small and variation under 2 Hz
            data.append([feOut[i], feSim[i], fiSim[i], adaptation[i]])
        i += 1
    data = np.array(data)

    if os.path.exists(name_file + '/P.npy'):
        # check if polynome is already compute
        p_with = np.load(name_file + '/P.npy')
        p_without = np.load(name_file + '/P_no_adpt.npy')
        TF = create_transfer_function(parameters_all, excitatory=excitatory)
    else:
        # Fitting data Zerlaut
        p_with, p_without, TF = fitting_model_zerlaut(data[:, 0], data[:, 1], data[:, 2], data[:, 3], parameters_all,
                                                      nb_value_fexc,
                                                      nb_value_finh, nb_value_adaptation,
                                                      MINadaptation, MAXadaptation, MINfinh, MAXfinh, MAXfexc,
                                                      excitatory)
        np.save(name_file + '/P.npy', p_with)
        np.save(name_file + '/P_no_adpt.npy', p_without)
    if print_details:
        print_result_box_plot(result_n,name_file_fig,nb_value_finh,nb_value_fexc)
        print_result_std(result_n,name_file_fig)
        print_result_curve_box_std(result_n, name_file_fig, nb_value_finh, nb_value_fexc)
    if print_error:
        print_result_zerlaut(result_n, TF, p_with, p_without, name_file_fig + 'zerlaut_', nb_value_finh,
                             nb_value_fexc)
    return p_with
