#  Copyright 2023 Aix-Marseille Université
# "Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements; and to You under the Apache License, Version 2.0. "
import numpy as np
import os
import sys
from parameter_analyse.check_derivation.helper_function import model, plot_derivation, \
    plot_derivation_check, plot_derivation_check_neg


def derivation_negative(params_all, path_P_excitatory, path_P_inhibitory, zeros_fe=-1.0e-9+1.0e-10, zeros_fi=1.0e-10,
                        external_fe=0.0, external_fi=0.0, min_scale=1e-10, max_cut=1e0,
                        ranges_excitatory=[np.linspace(0.0, 10.0, 1000) * 1e-3, np.linspace(0.0, 200.0, 1000) * 1e-3],
                        ranges_inhibitory=[np.linspace(0.0, 10.0, 1000) * 1e-3, np.linspace(0.0, 200.0, 1000) * 1e-3],
                        ranges_adaptation=[np.linspace(0.0, 100.0, 1000), np.linspace(0.0, 10000.0, 1000)],
                        save_name=''):
    """
    check the derivation can put the system in negative firing rate by computing the derivation for zeros
    :param params_all: parameters of the exploration
    :param path_P_excitatory: polynomial of excitatory neurons
    :param path_P_inhibitory: polynomial of inhibitory neurons
    :param zeros_fe: real zeros of firing rate
    :param zeros_fi: real zeros of firing rate
    :param external_fe: external input
    :param external_fi: external input
    :param min_scale: min of scale
    :param max_cut: maximum of cut for plotting
    :param ranges_excitatory: array of range of exploration of excitatory firing rate
    :param ranges_inhibitory: array of range of exploration of inhibitory firing rate
    :param ranges_adaptation: array of range of exploration of adaptation values
    :param save_name: add the name for saving
    :return:
    """
    Zerlaut_fit = model(params_all, path_P_excitatory, path_P_inhibitory)

    # excitatory Transfer function
    for i_inh, inh_values in enumerate(ranges_inhibitory):
        for i_adp, adaptation_values in enumerate(ranges_adaptation):
            TF = np.empty((inh_values.shape[0], adaptation_values.shape[0], 1))
            diff2_fe_fe = np.empty((inh_values.shape[0], adaptation_values.shape[0], 1))
            diff2_fi_fi = np.empty((inh_values.shape[0], adaptation_values.shape[0], 1))
            diff2_fe_fi = np.empty((inh_values.shape[0], adaptation_values.shape[0], 1))
            print(" index : ", end='')
            for index_inh, inh in enumerate(inh_values):
                for index_adpt, adpt in enumerate(adaptation_values):
                    print("\r index : ", index_inh, index_adpt, end=""); sys.stdout.flush()
                    TF[index_inh, index_adpt, :] = Zerlaut_fit.TF_excitatory(zeros_fe, inh, external_fe, external_fi, adpt)
                    diff2_fe_fe[index_inh, index_adpt, :] = Zerlaut_fit._diff2_fe_fe_e(zeros_fe, inh, external_fe,
                                                                                       external_fi, adpt)
                    diff2_fi_fi[index_inh, index_adpt, :] = Zerlaut_fit._diff2_fi_fi_e(zeros_fe, inh, external_fe,
                                                                                       external_fi, adpt)
                    diff2_fe_fi[index_inh, index_adpt, :] = Zerlaut_fit._diff2_fe_fi(Zerlaut_fit.TF_excitatory, zeros_fe,
                                                                                     inh, external_fe, external_fi, adpt)
            print()
            for name, data in [("transfer function", TF),
                               ("second order derivation fe", diff2_fe_fe),
                               ("second order derivation fi", diff2_fi_fi),
                               ("second order derivation fe and fi", diff2_fe_fi)]:
                np.save(save_name + "exc_" + str(i_inh) + "_" + str(i_adp) + "_" + name.replace(' ', '_') + ".npy",
                        data)
                # plot_derivation(save_name, name, i_inh, inh_values, i_adp, adaptation_values, data, type="exc")

    # inhibitory Transfer function
    for i_ex, ex_values in enumerate(ranges_excitatory):
        for i_adp, adaptation_values in enumerate(ranges_adaptation):
            TF = np.empty((ex_values.shape[0], adaptation_values.shape[0], 1))
            diff2_fe_fe = np.empty((ex_values.shape[0], adaptation_values.shape[0], 1))
            diff2_fi_fi = np.empty((ex_values.shape[0], adaptation_values.shape[0], 1))
            diff2_fe_fi = np.empty((ex_values.shape[0], adaptation_values.shape[0], 1))
            print(" index : ", end='')
            for index_ex, ex in enumerate(ex_values):
                for index_adpt, adpt in enumerate(adaptation_values):
                    print("\r index : ", index_ex, index_adpt, end=""); sys.stdout.flush()
                    TF[index_ex, index_adpt, :] = Zerlaut_fit.TF_inhibitory(ex, zeros_fi, external_fe, external_fi, adpt)
                    diff2_fe_fe[index_ex, index_adpt, :] = Zerlaut_fit._diff2_fe_fe_i(ex, zeros_fi, external_fe,
                                                                                      external_fi, adpt)
                    diff2_fi_fi[index_ex, index_adpt, :] = Zerlaut_fit._diff2_fi_fi_i(ex, zeros_fi, external_fe,
                                                                                      external_fi, adpt)
                    diff2_fe_fi[index_ex, index_adpt, :] = Zerlaut_fit._diff2_fe_fi(Zerlaut_fit.TF_inhibitory, ex,
                                                                                    zeros_fi, external_fe, external_fi,
                                                                                    adpt)
            print()
            for name, data in [("transfer function", TF),
                               ("second order derivation fe", diff2_fe_fe),
                               ("second order derivation fi", diff2_fi_fi),
                               ("second order derivation fe and fi", diff2_fe_fi)]:
                np.save(save_name + "inh_" + str(i_ex) + "_" + str(i_adp) + "_" + name.replace(' ', '_') + ".npy", data)
                # plot_derivation(save_name, name, i_ex, ex_values, i_adp, adaptation_values, data, type="exc")

    for i_adp, adaptation_values in enumerate(ranges_adaptation):
        for i_ex, ex_values in enumerate(ranges_excitatory):
            TF = np.load(save_name + "inh_" + str(i_ex) + "_" + str(i_adp) + "_transfer_function.npy")
            diff2_fe_fe = np.load(save_name + "inh_" + str(i_ex) + "_" + str(i_adp) + "_second_order_derivation_fe.npy")
            diff2_fi_fi = np.load(save_name + "inh_" + str(i_ex) + "_" + str(i_adp) + "_second_order_derivation_fi.npy")
            diff2_fe_fi = np.load(save_name + "inh_" + str(i_ex) + "_" + str(i_adp) + "_second_order_derivation_fe_and_fi.npy")
            print("TF sum ex : ", i_adp, i_ex, np.sum(TF < 0.0))
            print("diff: ", np.sum(np.logical_and(np.logical_and(diff2_fe_fe <= 0, diff2_fe_fi <= 0),
                                                  diff2_fi_fi <= 0)))
            index = np.unravel_index(np.argmin(TF + 0.5 * diff2_fe_fe + diff2_fe_fi + 0.5 * diff2_fi_fi), TF.shape)
            print("max: ", np.min(TF + 0.5 * diff2_fe_fe + diff2_fe_fi + 0.5 * diff2_fi_fi)*1e3, TF[index]*1e3)
            plot_derivation_check(save_name, i_ex, ex_values, i_adp, adaptation_values,
                                  TF, diff2_fe_fe, diff2_fi_fi, diff2_fe_fi,
                                  max_cut, min_scale, subtitle="inhibitory derivation function components",
                                  ylabel="excitatory firing rate in Hz", type='inhibitory')
            plot_derivation_check_neg(save_name, i_ex, ex_values, i_adp, adaptation_values,
                                      TF, diff2_fe_fe, diff2_fi_fi, diff2_fe_fi,
                                      max_cut, min_scale, subtitle="inhibitory derivation function components",
                                      ylabel="excitatory firing rate in Hz", type='inhibitory')

        for i_inh, inh_values in enumerate(ranges_inhibitory):
            TF = np.load(save_name + "exc_" + str(i_inh) + "_" + str(i_adp) + "_transfer_function.npy")
            diff2_fe_fe = np.load(save_name + "exc_" + str(i_inh) + "_" + str(i_adp) + "_second_order_derivation_fe.npy")
            diff2_fi_fi = np.load(save_name + "exc_" + str(i_inh) + "_" + str(i_adp) + "_second_order_derivation_fi.npy")
            diff2_fe_fi = np.load(save_name + "exc_" + str(i_inh) + "_" + str(i_adp) + "_second_order_derivation_fe_and_fi.npy")
            print("TF sum in : ", i_adp, i_inh, np.sum(TF < 0.0))
            print("diff: ", np.sum(np.logical_and(np.logical_and(diff2_fe_fe <= 0, diff2_fe_fi <= 0),
                                                  diff2_fi_fi <= 0)))
            index = np.unravel_index(np.argmin(TF + 0.5 * diff2_fe_fe + diff2_fe_fi + 0.5 * diff2_fi_fi), TF.shape)
            print("max: ", np.min(TF + 0.5 * diff2_fe_fe + diff2_fe_fi + 0.5 * diff2_fi_fi)*1e3, TF[index]*1e3)
            plot_derivation_check(save_name, i_inh, inh_values, i_adp, adaptation_values,
                                  TF, diff2_fe_fe, diff2_fi_fi, diff2_fe_fi,
                                  max_cut, min_scale, subtitle="excitatory derivation function components",
                                  ylabel="inhibitory firing rate in Hz", type='excitatory')
            plot_derivation_check_neg(save_name, i_inh, inh_values, i_adp, adaptation_values,
                                      TF, diff2_fe_fe, diff2_fi_fi, diff2_fe_fi,
                                      max_cut, min_scale, subtitle="excitatory derivation function components",
                                      ylabel="inhibitory firing rate in Hz", type='excitatory')


if __name__ == "__main__":
    from parameter_analyse.fitting_procedure.parameters import params_all
    path_P_excitatory = os.path.dirname(os.path.realpath(__file__)) + '/../fitting_procedure/fitting_50hz/C_m_200.0/t_ref_5.0/V_reset_-55.0/E_L_-63.0/g_L_10.0/I_e_0.0/a_0.0/b_0.0/Delta_T_2.0/tau_w_500.0/V_th_-50.0/E_ex_0.0/tau_syn_ex_5.0/E_in_-80.0/tau_syn_in_5.0/V_peak_0.0/N_tot_10000/p_connect_ex_0.05/p_connect_in_0.05/g_0.2/Q_e_1.5/Q_i_5.0/P.npy'
    path_P_inhibitory = os.path.dirname(os.path.realpath(__file__)) + '/../fitting_procedure/fitting_50hz/C_m_200.0/t_ref_5.0/V_reset_-65.0/E_L_-65.0/g_L_10.0/I_e_0.0/a_0.0/b_0.0/Delta_T_0.5/tau_w_1.0/V_th_-50.0/E_ex_0.0/tau_syn_ex_5.0/E_in_-80.0/tau_syn_in_5.0/V_peak_0.0/N_tot_10000/p_connect_ex_0.05/p_connect_in_0.05/g_0.2/Q_e_1.5/Q_i_5.0/P.npy'
    derivation_negative(params_all, path_P_excitatory, path_P_inhibitory)
    derivation_negative(params_all, path_P_excitatory, path_P_inhibitory,
                        ranges_excitatory=[np.linspace(0.0, 70.0, 1000) * 1e-3],
                        ranges_inhibitory=[],
                        ranges_adaptation=[np.linspace(0.0, 100.0, 1000), np.linspace(0.0, 10000.0, 1000)],
                        save_name='zoom'
                        )
    path_P_excitatory = os.path.dirname(os.path.realpath(__file__)) + '/../fitting_procedure/fitting_50hz/C_m_200.0/t_ref_5.0/V_reset_-55.0/E_L_-63.0/g_L_10.0/I_e_0.0/a_0.0/b_0.0/Delta_T_2.0/tau_w_500.0/V_th_-50.0/E_ex_0.0/tau_syn_ex_5.0/E_in_-80.0/tau_syn_in_5.0/V_peak_0.0/N_tot_10000/p_connect_ex_0.05/p_connect_in_0.05/g_0.2/Q_e_1.5/Q_i_5.0/P_no_adpt.npy'
    path_P_inhibitory = os.path.dirname(os.path.realpath(__file__)) + '/../fitting_procedure/fitting_50hz/C_m_200.0/t_ref_5.0/V_reset_-65.0/E_L_-65.0/g_L_10.0/I_e_0.0/a_0.0/b_0.0/Delta_T_0.5/tau_w_1.0/V_th_-50.0/E_ex_0.0/tau_syn_ex_5.0/E_in_-80.0/tau_syn_in_5.0/V_peak_0.0/N_tot_10000/p_connect_ex_0.05/p_connect_in_0.05/g_0.2/Q_e_1.5/Q_i_5.0/P_no_adpt.npy'
    derivation_negative(params_all, path_P_excitatory, path_P_inhibitory, save_name='no_adpt_')
    derivation_negative(params_all, path_P_excitatory, path_P_inhibitory, save_name='zoom_no_adpt_',
                        ranges_excitatory=[np.linspace(0.0, 1.0, 1000) * 1e-3],
                        ranges_inhibitory=[np.linspace(0.0, 33.0, 1000) * 1e-3, np.linspace(0.0, 1.0, 1000) * 1e-3],
                        ranges_adaptation=[np.linspace(0.0, 3500.0, 1000), np.linspace(0.0, 10000.0, 1000)],
                        )

