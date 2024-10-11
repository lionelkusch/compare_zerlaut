#  Copyright 2023 Aix-Marseille Université
# "Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements; and to You under the Apache License, Version 2.0. "
import numpy as np


def get_result_raw(name_file, MAXfout, MAXfexc, nb_value_fexc, nb_neurons,
                   MINfinh, MAXfinh, nb_value_finh, MINadaptation, MAXadaptation, nb_value_adaptation,
                   data_require=False):
    """
    get raw result of the firing rate
    :param name_file: folder of the result
    :param MAXfout: maximum of output firing rate
    :param MAXfexc: maximum excitatory input firing rate
    :param nb_value_fexc: number of different value of excitatory input firing rate
    :param nb_neurons: number of trial ofr each condition
    :param MAXfinh: maximum inhibitory input firing rate
    :param MINfinh: minimum inhibitory input firing rate
    :param nb_value_finh: number of different value of inhibitory input firing rate
    :param MAXadaptation: maximum of adaptation
    :param MINadaptation: minimum of adaptation
    :param nb_value_adaptation: number of different value of adaptation
    :param data_require: return data
    :return: result of the simulation
    """
    results = np.load(name_file + '/fout.npy').reshape(nb_value_adaptation * nb_value_fexc * nb_value_finh, nb_neurons)
    results *= 1e-3
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
    data = []
    while i != len(brut_fiSim):
        fi_model = np.where(brut_fiSim[i] == np.linspace(MINfinh, MAXfinh, nb_value_finh) * 1e-3)[0][0]
        w_model = np.where(brut_adaptation[i] == np.linspace(MINadaptation, MAXadaptation, nb_value_adaptation))[0][0]
        if brut_adaptation[i] < brut_adaptation[i - 1]:
            if brut_fiSim[i] < brut_fiSim[i - 1]:
                fe_model += 1
        result_n_brut[fe_model, fi_model, w_model, :6] = [brut_feOut[i], brut_feSim[i], brut_fiSim[i],
                                                          brut_adaptation[i], brut_feOut_std[i], brut_feOut_med[i]]
        result_n_brut[fe_model, fi_model, w_model, 6:] = results[i]
        if not ((brut_feOut[i] > MAXfout * 1e-3) and (
                brut_feSim[i] > MAXfexc)) and data_require:  # coefficient of variation small and variation under 2 Hz
            data.append([brut_feOut[i], brut_feSim[i], brut_fiSim[i], brut_adaptation[i]])
        i += 1
    if data_require:
        return result_n_brut, np.array(data)
    else:
        return result_n_brut


def remove_outlier(datas, p=3):
    """
    remove outlier by removing value of the extreme case based on normal distribution
    :param datas: data for removing values
    :param p: number of standard deviation used for removing outlier
    :return:
    """
    Q1, Q3 = np.quantile(datas, q=[0.25, 0.75], axis=1)
    IQR = Q3 - Q1
    min_data, max_data = Q1 - p * IQR, Q3 + p * IQR
    result = np.empty(datas.shape)
    result[:] = np.NAN
    for i, data in enumerate(datas):
        data_pre = data[np.logical_and(min_data[i] <= data, data <= max_data[i])]
        result[i, :data_pre.shape[0]] = data_pre
    return result


def get_result(name_file, MAXfout, MAXfexc, nb_value_fexc, nb_neurons,
               MINfinh, MAXfinh, nb_value_finh, MINadaptation, MAXadaptation, nb_value_adaptation, data_require):
    """
    get raw result of the firing rate
    :param name_file: folder of the result
    :param MAXfexc: maximum excitatory input firing rate
    :param MAXfout: maximum of output firing rate
    :param nb_value_fexc: number of different value of excitatory input firing rate
    :param nb_neurons: number of trial ofr each condition
    :param MAXfinh: maximum inhibitory input firing rate
    :param MINfinh: minimum inhibitory input firing rate
    :param nb_value_finh: number of different value of inhibitory input firing rate
    :param MAXadaptation: maximum of adaptation
    :param MINadaptation: minimum of adaptation
    :param nb_value_adaptation: number of different value of adaptation
    :param data_require: return data
    :return: result of the simulation without outlier
    """
    results = np.load(name_file + '/fout.npy').reshape(nb_value_adaptation * nb_value_fexc * nb_value_finh, nb_neurons)
    results *= 1e-3
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

    # shape data and remove fe higher than some values
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
        if ((feOut[i] <= MAXfout * 1e-3) and (feSim[i] <= MAXfexc * 1e-3)) and data_require:  # coefficient of variation small and variation under 2 Hz
            data.append([feOut[i], feSim[i], fiSim[i], adaptation[i]])
        i += 1
    if data_require:
        return result_n, np.array(data)
    else:
        return result_n
