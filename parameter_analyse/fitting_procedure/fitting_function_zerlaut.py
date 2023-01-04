#  Copyright 2021 Aix-Marseille Université
# "Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements; and to You under the Apache License, Version 2.0. "
import numpy as np
import scipy.special as sp_spec
from scipy.optimize import minimize
from .Zerlaut import ZerlautAdaptationSecondOrder as model


def create_transfer_function(parameter, excitatory):
    """
    create the transfer function from the model of Zerlaut adapted for inhibitory and excitatory neurons
    :param parameter: parameter of the simulation
    :param excitatory: if for excitatory or inhibitory population
    :return: the transfer function
    """
    model_test = model()
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
    model_test.external_input_excitatory_to_excitatory = np.array(0.0)
    model_test.external_input_excitatory_to_inhibitory = np.array(0.0)
    model_test.external_input_inhibitory_to_excitatory = np.array(0.0)
    model_test.external_input_inhibitory_to_inhibitory = np.array(0.0)
    model_test.K_ext_e = np.array(0)
    model_test.K_ext_i = np.array(0)
    if excitatory:
        def TF(fe, fi, p, f_ext_e=0.0, f_ext_i=0.0, w=0.0):
            model_test.P_e = p
            return model_test.TF_excitatory(fe, fi, f_ext_e, f_ext_i, w)
    else:
        def TF(fe, fi, p, f_ext_e=0.0, f_ext_i=0.0, w=0.0):
            model_test.P_i = p
            return model_test.TF_inhibitory(fe, fi, f_ext_e, f_ext_i, w)
    return TF


def effective_Vthre(rate, muV, sV, Tv):
    """
    effective of voltage membrane
    :param rate: firing rate
    :param muV: mean voltage
    :param sV: std of voltage
    :param Tv: time constant
    :return:
    """
    Vthre_eff = muV + np.sqrt(2) * sV * sp_spec.erfcinv(rate * 2. * Tv)  # effective threshold
    return Vthre_eff


def fit_data(feOut, feSim, fiSim, adaptation, TF, parameters, excitatory):
    """
    fit result for finding the polynomial
    :param feOut: firing output result
    :param feSim: input excitatory firing rate
    :param fiSim: output inhibitory firing rate
    :param adaptation: adaptation value
    :param TF: transfer function
    :param parameters: parameters of the models
    :param excitatory: excitatory or inhibitory neurons
    :return:
    """
    # Compute mean of value for the model
    E_L = parameters['E_L_e'] if excitatory else parameters['E_L_i']
    muV, sV, Tv = model.get_fluct_regime_vars(feSim, fiSim, 0.0, 0.0, adaptation, parameters['Q_e'],
                                              parameters['tau_syn_ex'], parameters['E_ex'], parameters['Q_i'],
                                              parameters['tau_syn_in'], parameters['E_in'],
                                              parameters['g_L'], parameters['C_m'], E_L,
                                              parameters['N_tot'], parameters['p_connect_ex'],
                                              parameters['p_connect_in'], parameters['g'], 0.0, 0.0)
    Tv += parameters['g_L'] / parameters['C_m']
    i_non_zeros = np.where(feOut * Tv < 1.0)
    Vthre_eff = effective_Vthre(feOut[i_non_zeros], muV[i_non_zeros], sV[i_non_zeros], Tv[i_non_zeros]) * 1e-3
    TvN = Tv[i_non_zeros] * parameters['g_L'] / parameters['C_m']

    # initialisation of the fitting
    P = np.zeros(10)
    P[:5] = Vthre_eff.mean(), 1e-3, 1e-3, 1e-3, 1e-3

    # fitting the voltage threshold
    def Res(p):
        """
        absolute error
        :param p: polynomial
        :return:
        """
        pp = p
        vthre = model.threshold_func(muV[i_non_zeros], sV[i_non_zeros], TvN, *pp)
        return np.mean((Vthre_eff - vthre) ** 2)
    plsq = minimize(Res, P, method='SLSQP', options={'ftol': 1e-15, 'disp': True, 'maxiter': 50000})
    P = plsq.x
    print("error ", np.mean(((feOut - TF(feSim, fiSim, P, w=adaptation)) ** 2) * 1e3))

    # fitting the mean firing rate
    def Res_2(p):
        """
        absolute error
        :param p: polynomial
        :return:
        """
        return np.mean(((feOut - TF(feSim, fiSim, p, w=adaptation)) * 1e3) ** 2)
    plsq = minimize(Res_2, P, method='nelder-mead', tol=1e-11, options={'xatol': 1e-15, 'disp': True, 'maxiter': 50000})
    return plsq.x


def fitting_model_zerlaut(feOut, feSim, fiSim, adaptation, parameters, excitatory, print_result, save_result=None,
                          fitting=True):
    """
    fit the model with our without adaptation
    :param feOut: firing output result
    :param feSim: input excitatory firing rate
    :param fiSim: output inhibitory firing rate
    :param adaptation: adaptation value
    :param parameters: parameters of the models
    :param excitatory: excitatory or inhibitory neurons
    :param print_result: print result of the fitting
    :param save_result: save result
    :return:
    """
    mask = np.where(adaptation == 0.0)
    feOut_1 = feOut[mask]
    feSim_1 = feSim[mask]
    fiSim_1 = fiSim[mask]
    adaptation_1 = adaptation[mask]

    TF = create_transfer_function(parameters, excitatory=excitatory)
    if fitting:
        p_with = fit_data(feOut, feSim, fiSim, adaptation, TF, parameters, excitatory)
        p_without = fit_data(feOut_1, feSim_1, fiSim_1, adaptation_1, TF, parameters, excitatory)
    else:
        p_with = np.load(save_result + '/P.npy')
        p_without = np.load(save_result + '/P_no_adpt.npy')

    if print_result:
        np.set_printoptions(edgeitems=3, infstr='inf', linewidth=75, nanstr='nan', precision=8, suppress=False,
                            threshold=1000, formatter=None)

        print("######################## fitting without adaptation ######################")
        print("                    #### data without adaptation    #####                 ")
        index = np.argsort(np.abs(((feOut_1 - TF(feSim_1, fiSim_1, p_without, w=adaptation_1)) * 1e3)))[-5:]
        print("frequency ex", feSim_1[index] * 1e3)
        print("frequency in", fiSim_1[index] * 1e3)
        print("adaptation", adaptation_1[index])
        print("expected : ", feOut_1[index] * 1e3)
        print("got : ", TF(feSim_1, fiSim_1, p_without, w=adaptation_1)[index] * 1e3)
        print("error : ", np.abs(((feOut_1 - TF(feSim_1, fiSim_1, p_without, w=adaptation_1)) * 1e3))[index])
        print("max error ", np.abs(((feOut_1 - TF(feSim_1, fiSim_1, p_without, w=adaptation_1)) * 1e3))[index[-1]])
        print("error ", np.mean(((feOut_1 - TF(feSim_1, fiSim_1, p_without, w=adaptation_1)) * 1e3) ** 2))
        print("error relative ", np.mean(((feOut_1 - TF(feSim_1, fiSim_1, p_without, w=adaptation_1)) / feOut_1) ** 2))
        print("                    #### all data                   ####                  ")
        index = np.argsort(np.abs(((feOut - TF(feSim, fiSim, p_without, w=adaptation)) * 1e3)))[-5:]
        print("frequency ex", feSim[index] * 1e3)
        print("frequency in", fiSim[index] * 1e3)
        print("adaptation", adaptation[index])
        print("expected : ", feOut[index] * 1e3)
        print("got : ", TF(feSim, fiSim, p_without, w=adaptation)[index] * 1e3)
        print("error : ", np.abs(((feOut - TF(feSim, fiSim, p_without, w=adaptation)) * 1e3))[index])
        print("max error ", np.abs(((feOut - TF(feSim, fiSim, p_without, w=adaptation)) * 1e3))[index[-1]])
        print("error ", np.mean(((feOut - TF(feSim, fiSim, p_without, w=adaptation)) * 1e3) ** 2))
        print("error relative ", np.mean(((feOut - TF(feSim, fiSim, p_without, w=adaptation)) / feOut) ** 2))
        print("##########################################################################")
        print(p_without)
        print("######################## fitting with adaptation    ######################")
        print("                    #### data without adaptation    #####                 ")
        index = np.argsort(np.abs(((feOut_1 - TF(feSim_1, fiSim_1, p_with, w=adaptation_1)) * 1e3)))[-5:]
        print("frequency ex", feSim_1[index] * 1e3)
        print("frequency in", fiSim_1[index] * 1e3)
        print("adaptation", adaptation_1[index])
        print("expected : ", feOut_1[index] * 1e3)
        print("got : ", TF(feSim_1, fiSim_1, p_with, w=adaptation_1)[index] * 1e3)
        print("error : ", np.abs(((feOut_1 - TF(feSim_1, fiSim_1, p_with, w=adaptation_1)) * 1e3))[index])
        print("max error ", np.abs(((feOut_1 - TF(feSim_1, fiSim_1, p_with, w=adaptation_1)) * 1e3))[index[-1]])
        print("error ", np.mean(((feOut_1 - TF(feSim_1, fiSim_1, p_with, w=adaptation_1)) * 1e3) ** 2))
        print("error relative ", np.mean(((feOut_1 - TF(feSim_1, fiSim_1, p_with, w=adaptation_1)) / feOut_1) ** 2))
        print("                    #### all data                   ####                  ")
        index = np.argsort(np.abs(((feOut - TF(feSim, fiSim, p_with, w=adaptation)) * 1e3)))[-5:]
        print("frequency ex", feSim[index] * 1e3)
        print("frequency in", fiSim[index] * 1e3)
        print("adaptation", adaptation[index])
        print("expected : ", feOut[index] * 1e3)
        print("got : ", TF(feSim, fiSim, p_with, w=adaptation)[index] * 1e3)
        print("error : ", np.abs(((feOut - TF(feSim, fiSim, p_with, w=adaptation)) * 1e3))[index])
        print("max error ", np.abs(((feOut - TF(feSim, fiSim, p_with, w=adaptation)) * 1e3))[index[-1]])
        print("error ", np.mean(((feOut - TF(feSim, fiSim, p_with, w=adaptation)) * 1e3) ** 2))
        print("error relative ", np.mean(((feOut - TF(feSim, fiSim, p_with, w=adaptation)) / feOut) ** 2))
        print("##########################################################################")
        print(p_with)
    if save_result is not None and fitting:
        np.save(save_result + '/P.npy', p_with)
        np.save(save_result + '/P_no_adpt.npy', p_without)

    return p_with, p_without, TF
