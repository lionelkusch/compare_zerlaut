from parameter_analyse.Zerlaut import ZerlautAdaptationSecondOrder
from parameter_analyse.parameters import params_all
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm, LogNorm, Normalize
from matplotlib import cm


min_scale = 1e-7
max_cut = 1e0
cmap = cm.get_cmap('viridis')

def model(parameter, path_TF_e, path_TF_i):
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
    def _diff_fe(TF, fe, fi, fe_ext, fi_ext, W, df=1e-7):
        return (TF(fe + df, fi, fe_ext, fi_ext, W) - TF(fe - df, fi, fe_ext, fi_ext, W)) / (2 * df * 1e3)

    model_test._diff_fe = _diff_fe

    def _diff_fi(TF, fe, fi, fe_ext, fi_ext, W, df=1e-7):
        return (TF(fe, fi + df, fe_ext, fi_ext, W) - TF(fe, fi - df, fe_ext, fi_ext, W)) / (2 * df * 1e3)

    model_test._diff_fi = _diff_fi

    def _diff2_fe_fe_e(fe, fi, fe_ext, fi_ext, W, df=1e-7):
        TF = model_test.TF_excitatory
        return (TF(fe + df, fi, fe_ext, fi_ext, W) - 2 * TF(fe, fi, fe_ext, fi_ext, W) + TF(fe - df, fi, fe_ext, fi_ext,
                                                                                            W)) / ((df * 1e3) ** 2)

    model_test._diff2_fe_fe_e = _diff2_fe_fe_e

    def _diff2_fe_fe_i(fe, fi, fe_ext, fi_ext, W, df=1e-7):
        TF = model_test.TF_inhibitory
        return (TF(fe + df, fi, fe_ext, fi_ext, W) - 2 * TF(fe, fi, fe_ext, fi_ext, W) + TF(fe - df, fi, fe_ext, fi_ext,
                                                                                            W)) / ((df * 1e3) ** 2)

    model_test._diff2_fe_fe_i = _diff2_fe_fe_i

    def _diff2_fi_fe(TF, fe, fi, fe_ext, fi_ext, W, df=1e-7):
        return (_diff_fi(TF, fe + df, fi, fe_ext, fi_ext, W) - _diff_fi(TF, fe - df, fi, fe_ext, fi_ext, W)) / (
                    2 * df * 1e3)

    model_test._diff2_fi_fe = _diff2_fi_fe

    def _diff2_fe_fi(TF, fe, fi, fe_ext, fi_ext, W, df=1e-7):
        return (_diff_fe(TF, fe, fi + df, fe_ext, fi_ext, W) - _diff_fe(TF, fe, fi - df, fe_ext, fi_ext, W)) / (
                    2 * df * 1e3)

    model_test._diff2_fe_fi = _diff2_fe_fi

    def _diff2_fi_fi_e(fe, fi, fe_ext, fi_ext, W, df=1e-7):
        TF = model_test.TF_excitatory
        return (TF(fe, fi + df, fe_ext, fi_ext, W) - 2 * TF(fe, fi, fe_ext, fi_ext, W) + TF(fe, fi - df, fe_ext, fi_ext,
                                                                                            W)) / ((df * 1e3) ** 2)

    model_test._diff2_fi_fi_e = _diff2_fi_fi_e

    def _diff2_fi_fi_i(fe, fi, fe_ext, fi_ext, W, df=1e-7):
        TF = model_test.TF_inhibitory
        return (TF(fe, fi + df, fe_ext, fi_ext, W) - 2 * TF(fe, fi, fe_ext, fi_ext, W) + TF(fe, fi - df, fe_ext, fi_ext,
                                                                                            W)) / ((df * 1e3) ** 2)

    model_test._diff2_fi_fi_i = _diff2_fi_fi_i

    return model_test


Zerlaut_fit = model(params_all,
                    '/home/kusch/Documents/project/Zerlaut/compare_zerlaut/parameter_analyse/fitting/C_m_200.0/t_ref_5.0/V_reset_-55.0/E_L_-63.0/g_L_10.0/I_e_0.0/a_0.0/b_0.0/Delta_T_2.0/tau_w_500.0/V_th_-50.0/E_ex_0.0/tau_syn_ex_5.0/E_in_-80.0/tau_syn_in_5.0/V_peak_0.0/N_tot_10000/p_connect_ex_0.05/p_connect_in_0.05/g_0.2/Q_e_1.5/Q_i_5.0/P.npy',
                    '/home/kusch/Documents/project/Zerlaut/compare_zerlaut/parameter_analyse/fitting/C_m_200.0/t_ref_5.0/V_reset_-65.0/E_L_-65.0/g_L_10.0/I_e_0.0/a_0.0/b_0.0/Delta_T_0.5/tau_w_1.0/V_th_-50.0/E_ex_0.0/tau_syn_ex_5.0/E_in_-80.0/tau_syn_in_5.0/V_peak_0.0/N_tot_10000/p_connect_ex_0.05/p_connect_in_0.05/g_0.2/Q_e_1.5/Q_i_5.0/P.npy')

# zeros_fe = - 1.0e-6
# zeros_fi = 0.0
# external_fe = 0.0
# external_fi = 0.0
# # excitatory Transfer function
# for i_inh, inh_values in enumerate([np.linspace(0.0, 200.0, 1000) * 1e-3, np.linspace(0.0, 10.0, 1000) * 1e-3]):
#     for i_adp, adaptation_values in enumerate([np.linspace(0.0, 10.0, 1000)]):#, np.linspace(0.0, 10000.0, 1000)]):
#         TF = np.empty((inh_values.shape[0], adaptation_values.shape[0], 1))
#         diff2_fe_fe = np.empty((inh_values.shape[0], adaptation_values.shape[0], 1))
#         diff2_fi_fi = np.empty((inh_values.shape[0], adaptation_values.shape[0], 1))
#         diff2_fe_fi = np.empty((inh_values.shape[0], adaptation_values.shape[0], 1))
#         print(" index : ")
#         for index_inh, inh in enumerate(inh_values):
#             for index_adpt, adpt in enumerate(adaptation_values):
#                 print("\r index : ",index_inh,index_adpt,end=""); sys.stdout.flush()
#                 TF[index_inh, index_adpt, :]=Zerlaut_fit.TF_excitatory(zeros_fe, inh, external_fe, external_fi, adpt)
#                 diff2_fe_fe[index_inh, index_adpt, :]=Zerlaut_fit._diff2_fe_fe_e(zeros_fe, inh, external_fe, external_fi, adpt)
#                 diff2_fi_fi[index_inh, index_adpt, :]=Zerlaut_fit._diff2_fi_fi_e(zeros_fe, inh, external_fe, external_fi, adpt)
#                 diff2_fe_fi[index_inh, index_adpt, :]=Zerlaut_fit._diff2_fe_fi(Zerlaut_fit.TF_excitatory,zeros_fe, inh, external_fe, external_fi, adpt)
#         for name, data in [("transfer function", TF),
#                            ("second order derivation fe", diff2_fe_fe),
#                            ("second order derivation fi", diff2_fi_fi),
#                            ("second order derivation fe and fi", diff2_fe_fi)]:
#             np.save("exc_"+str(i_inh)+"_"+str(i_adp)+"_"+name+".npy", data)
#             if np.where(np.isnan(data))[0].shape[0] != 0:
#                 data[np.where(np.isnan(data))] = 0.0
#                 print(name,inh_values[np.where(np.isnan(data))[0]], adaptation_values[np.where(np.isnan(data))[0]])
#             fig = plt.figure(figsize=(20,20))
#             ax = fig.add_subplot(projection='3d')
#             X, Y = np.meshgrid(adaptation_values, inh_values)
#             Z = data[:, :, 0]
#             surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, rcount=1000,
#                             vmin=-np.max(np.abs((np.nanmax(Z), np.nanmin(Z)))), vmax=np.max(np.abs((np.nanmax(Z), np.nanmin(Z)))))
#             fig.colorbar(surf, shrink=0.5, aspect=10)
#             plt.title(name)
#             plt.savefig("exc_"+str(i_inh)+"_"+str(i_adp)+"_"+name)
#             if np.nanmin(Z) < 0:
#                 fig = plt.figure(figsize=(20,20))
#                 ax = fig.add_subplot(projection='3d')
#                 X, Y = np.meshgrid(adaptation_values, inh_values)
#                 Z = data[:, :, 0]
#                 surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, rcount=1000,
#                                 vmin=np.nanmin(Z), vmax=0.0)
#                 ax.set_zlim(zmax=0.1)
#                 surf.cmap.set_over('w')
#                 fig.colorbar(surf, shrink=0.5, aspect=10)
#                 plt.title(name+' negative')
#             else:
#                 print("no negative values  for : ",name)
#                 plt.figure(figsize=(20,20))
#             plt.savefig("neg_exc_"+str(i_inh)+"_"+str(i_adp)+"_"+name)
#
# # inhibitory Transfer function
# for i_ex, ex_values in enumerate([np.linspace(0.0, 200.0, 1000) * 1e-3, np.linspace(0.0, 10.0, 1000) * 1e-3]):
#     for i_adp, adaptation_values in enumerate([np.linspace(0.0, 10.0, 1000)]):#, np.linspace(0.0, 1000.0, 1000)]):
#         TF = np.empty((ex_values.shape[0], adaptation_values.shape[0], 1))
#         diff2_fe_fe = np.empty((ex_values.shape[0], adaptation_values.shape[0], 1))
#         diff2_fi_fi = np.empty((ex_values.shape[0], adaptation_values.shape[0], 1))
#         diff2_fe_fi = np.empty((ex_values.shape[0], adaptation_values.shape[0], 1))
#         for index_ex, ex in enumerate(ex_values):
#             for index_adpt, adpt in enumerate(adaptation_values):
#                 print("\r index : ",index_ex, index_adpt,end=""); sys.stdout.flush()
#                 TF[index_ex, index_adpt, :] = Zerlaut_fit.TF_inhibitory(ex, zeros_fi, external_fe, external_fi, adpt)
#                 diff2_fe_fe[index_ex, index_adpt, :] = Zerlaut_fit._diff2_fe_fe_i(ex, zeros_fi, external_fe,
#                                                                                    external_fi, adpt)
#                 diff2_fi_fi[index_ex, index_adpt, :] = Zerlaut_fit._diff2_fi_fi_i(ex, zeros_fi, external_fe,
#                                                                                    external_fi, adpt)
#                 diff2_fe_fi[index_ex, index_adpt, :] = Zerlaut_fit._diff2_fe_fi(Zerlaut_fit.TF_inhibitory, ex,
#                                                                                  zeros_fi, external_fe, external_fi, adpt)
#         for name, data in [("transfer function", TF),
#                            ("second order derivation fe", diff2_fe_fe),
#                            ("second order derivation fi", diff2_fi_fi),
#                            ("second order derivation fe and fi", diff2_fe_fi)]:
#             np.save("inh_"+str(i_ex)+"_"+str(i_adp)+"_"+name+".npy", data)
#             if np.where(np.isnan(data))[0].shape[0] != 0:
#                 data[np.where(np.isnan(data))] = 0.0
#                 print(name, ex_values[np.where(np.isnan(data))[0]], adaptation_values[np.where(np.isnan(data))[0]])
#             fig = plt.figure()
#             ax = fig.add_subplot(projection='3d')
#             X, Y = np.meshgrid(adaptation_values, ex_values)
#             Z = data[:, :, 0]
#             surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, rcount=1000,
#                                    vmin=-np.max(np.abs((np.nanmax(Z), np.nanmin(Z)))),
#                                    vmax=np.max(np.abs((np.nanmax(Z), np.nanmin(Z)))))
#             fig.colorbar(surf, shrink=0.5, aspect=10)
#             plt.title(name)
#             plt.savefig("inh_" + str(i_ex) + "_" + str(i_adp) + "_" + name)
#             if np.nanmin(Z) < 0:
#                 fig = plt.figure(figsize=(20,20))
#                 ax = fig.add_subplot(projection='3d')
#                 X, Y = np.meshgrid(adaptation_values, ex_values)
#                 Z = data[:, :, 0]
#                 surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, rcount=1000,
#                                        vmin=np.nanmin(Z), vmax=0.0)
#                 ax.set_zlim(zmax=0.1)
#                 surf.cmap.set_over('w')
#                 fig.colorbar(surf, shrink=0.5, aspect=10)
#                 plt.title(name + ' negative')
#             else:
#                 plt.figure(figsize=(20,20))
#                 print("no negative values  for : ", name)
#             plt.savefig("neg_inh_" + str(i_ex) + "_" + str(i_adp) + "_" + name)


for i_adp, adaptation_values in enumerate([np.linspace(0.0, 1000.0, 1000), np.linspace(0.0, 10.0, 1000)]):
    for i_ex, ex_values in enumerate([np.linspace(0.0, 200.0, 1000) * 1e-3, np.linspace(0.0, 10.0, 1000) * 1e-3]):
        if os.path.exists("inh_" + str(i_ex) + "_" + str(i_adp) + "_" + "transfer function" + ".npy"):
            TF = np.load("inh_" + str(i_ex) + "_" + str(i_adp) + "_" + "transfer function" + ".npy") * 1e3
            diff2_fe_fe = np.load("inh_" + str(i_ex) + "_" + str(i_adp) + "_" + "second order derivation fe" + ".npy")
            diff2_fi_fi = np.load("inh_" + str(i_ex) + "_" + str(i_adp) + "_" + "second order derivation fi" + ".npy")
            diff2_fe_fi = np.load("inh_" + str(i_ex) + "_" + str(i_adp) + "_" + "second order derivation fe and fi" + ".npy")

            fig, axs = plt.subplots(2, 2, figsize=(20, 20))
            fig.suptitle("inhibitory derivation function components")

            max_limit = np.nanmax(TF) if np.nanmax(TF) < max_cut*1e3 else max_cut*1e3
            min_limit = np.nanmin(TF) if -np.nanmin(TF) < max_cut*1e3 else -max_cut*1e3
            if np.nanmin(TF) >= 0.0:
                im_0 = axs[0, 0].imshow(TF, norm=Normalize(vmin=min_limit, vmax=max_limit), cmap=cmap)
            else:
                linthresh = np.min([np.nanmax(TF), -np.nanmin(TF)]) * min_scale
                im_0 = axs[0, 0].imshow(TF, norm=SymLogNorm(linthresh=linthresh, vmin=min_limit, vmax=max_limit), cmap=cmap)
            fig.colorbar(im_0, ax=axs[0, 0])
            ticks_positions = np.array(np.around(np.linspace(0, ex_values.shape[0] - 1, 7)), dtype=int)
            axs[0, 0].set_yticks(ticks_positions)
            axs[0, 0].set_yticklabels(np.around(ex_values[ticks_positions], decimals=3) * 1e3)
            axs[0, 0].set_ylabel("excitatory firing rate in Hz")
            ticks_positions = np.array(np.around(np.linspace(0, adaptation_values.shape[0] - 1, 7)), dtype=int)
            axs[0, 0].set_xticks(ticks_positions)
            axs[0, 0].set_xticklabels(np.around(adaptation_values[ticks_positions], decimals=0))
            axs[0, 0].set_xlabel("adaptation")
            axs[0, 0].set_title('Transfer function of inhibitory neurons in Hz')
            if np.nanmin(TF) < 0.0:
                axs[0, 0].contourf(np.arange(0, TF.shape[0], 1),
                                   np.arange(0, TF.shape[1], 1),
                                   TF[:, :, 0], levels=[-10000.0, 0.0],
                                   hatches=['/'], colors=['red'], alpha=0.0)
                axs[0, 0].contour(np.arange(0, TF.shape[0], 1),
                                  np.arange(0, TF.shape[1], 1),
                                  TF[:, :, 0], levels=[0.0], colors=['black'])

            max_limit = np.nanmax(diff2_fe_fe) if np.nanmax(diff2_fe_fe) < max_cut else max_cut
            min_limit = np.nanmin(diff2_fe_fe) if -np.nanmin(diff2_fe_fe) < max_cut else -max_cut
            linthresh = np.nanmin(diff2_fe_fe) if np.nanmin(diff2_fe_fe) > 0.0 else np.min([np.nanmax(diff2_fe_fe), -np.nanmin(diff2_fe_fe)]) * min_scale
            im_1 = axs[0, 1].imshow(diff2_fe_fe, norm=SymLogNorm(linthresh=linthresh, vmin=min_limit, vmax=max_limit), cmap=cmap)
            fig.colorbar(im_1, ax=axs[0, 1])
            axs[0, 1].set_title('$\partial^2f_e/\partial^2f$ TF')
            ticks_positions = np.array(np.around(np.linspace(0, ex_values.shape[0] - 1, 7)), dtype=int)
            axs[0, 1].set_yticks(ticks_positions)
            axs[0, 1].set_yticklabels(np.around(ex_values[ticks_positions], decimals=3) * 1e3)
            axs[0, 1].set_ylabel("excitatory firing rate in Hz")
            ticks_positions = np.array(np.around(np.linspace(0, adaptation_values.shape[0] - 1, 7)), dtype=int)
            axs[0, 1].set_xticks(ticks_positions)
            axs[0, 1].set_xticklabels(np.around(adaptation_values[ticks_positions], decimals=0))
            axs[0, 1].set_xlabel("adaptation")
            if np.nanmin(diff2_fe_fe) < 0.0:
                axs[0, 1].contourf(np.arange(0, diff2_fe_fe.shape[0], 1),
                                   np.arange(0, diff2_fe_fe.shape[1], 1),
                                   diff2_fe_fe[:, :, 0], levels=[-10000.0, 0.0],
                                   hatches=['/'], colors=['red'], alpha=0.0)
                axs[0, 1].contour(np.arange(0, diff2_fe_fe.shape[0], 1),
                                  np.arange(0, diff2_fe_fe.shape[1], 1),
                                  diff2_fe_fe[:, :, 0], levels=[0.0], colors=['black'])

            max_limit = np.nanmax(diff2_fi_fi) if np.nanmax(diff2_fi_fi) < max_cut else max_cut
            min_limit = np.nanmin(diff2_fi_fi) if -np.nanmin(diff2_fi_fi) < max_cut else -max_cut
            linthresh = np.nanmin(diff2_fi_fi) if np.nanmin(diff2_fi_fi) > 0.0 else np.min([np.nanmax(diff2_fi_fi), -np.nanmin(diff2_fi_fi)]) * min_scale
            im_2 = axs[1, 0].imshow(diff2_fi_fi, norm=SymLogNorm(linthresh=linthresh, vmin=min_limit, vmax=max_limit), cmap=cmap)
            fig.colorbar(im_2, ax=axs[1, 0])
            axs[1, 0].set_title('$\partial^2f_i/\partial^2f$ TF')
            ticks_positions = np.array(np.around(np.linspace(0, ex_values.shape[0] - 1, 7)), dtype=int)
            axs[1, 0].set_yticks(ticks_positions)
            axs[1, 0].set_yticklabels(np.around(ex_values[ticks_positions], decimals=3) * 1e3)
            axs[1, 0].set_ylabel("excitatory firing rate in Hz")
            ticks_positions = np.array(np.around(np.linspace(0, adaptation_values.shape[0] - 1, 7)), dtype=int)
            axs[1, 0].set_xticks(ticks_positions)
            axs[1, 0].set_xticklabels(np.around(adaptation_values[ticks_positions], decimals=0))
            axs[1, 0].set_xlabel("adaptation")
            if np.nanmin(diff2_fi_fi) < 0.0:
                axs[1, 0].contourf(np.arange(0, diff2_fi_fi.shape[0], 1),
                                   np.arange(0, diff2_fi_fi.shape[1], 1),
                                   diff2_fi_fi[:, :, 0], levels=[-10000.0, 0.0],
                                   hatches=['/'], colors=['red'], alpha=0.0)
                axs[1, 0].contour(np.arange(0, diff2_fi_fi.shape[0], 1),
                                  np.arange(0, diff2_fi_fi.shape[1], 1),
                                  diff2_fi_fi[:, :, 0], levels=[0.0], colors=['black'])

            max_limit = np.nanmax(diff2_fe_fi) if np.nanmax(diff2_fe_fi) < max_cut else max_cut
            min_limit = np.nanmin(diff2_fe_fi) if -np.nanmin(diff2_fe_fi) < max_cut else -max_cut
            linthresh = np.nanmin(diff2_fe_fi) if np.nanmin(diff2_fe_fi) > 0.0 else np.min([np.nanmax(diff2_fe_fi), -np.nanmin(diff2_fe_fi)]) * min_scale
            im_3 = axs[1, 1].imshow(diff2_fe_fi, norm=SymLogNorm(linthresh=linthresh, vmin=min_limit, vmax=max_limit), cmap=cmap)
            fig.colorbar(im_3, ax=axs[1, 1])
            ticks_positions = np.array(np.around(np.linspace(0, ex_values.shape[0] - 1, 7)), dtype=int)
            axs[1, 1].set_yticks(ticks_positions)
            axs[1, 1].set_yticklabels(np.around(ex_values[ticks_positions], decimals=3) * 1e3)
            axs[1, 1].set_ylabel("excitatory firing rate in Hz")
            ticks_positions = np.array(np.around(np.linspace(0, adaptation_values.shape[0] - 1, 7)), dtype=int)
            axs[1, 1].set_xticks(ticks_positions)
            axs[1, 1].set_xticklabels(np.around(adaptation_values[ticks_positions], decimals=0))
            axs[1, 1].set_xlabel("adaptation")
            axs[1, 1].set_title('$\partial f_i\partial f_e/\partial^2f$ TF')
            if np.nanmin(diff2_fe_fi) < 0.0:
                axs[1, 1].contourf(np.arange(0, diff2_fe_fi.shape[0], 1),
                                   np.arange(0, diff2_fe_fi.shape[1], 1),
                                   diff2_fe_fi[:, :, 0], levels=[-10000.0, 0.0],
                                   hatches=['/'], colors=['red'], alpha=0.0)
                axs[1, 1].contour(np.arange(0, diff2_fe_fi.shape[0], 1),
                                  np.arange(0, diff2_fe_fi.shape[1], 1),
                                  diff2_fe_fi[:, :, 0], levels=[0.0], colors=['black'])
            plt.savefig("inh_" + str(i_ex) + "_" + str(i_adp)+'.svg')

    for i_inh, inh_values in enumerate([np.linspace(0.0, 200.0, 1000) * 1e-3, np.linspace(0.0, 10.0, 1000) * 1e-3]):
        if os.path.exists("exc_" + str(i_inh) + "_" + str(i_adp) + "_" + "transfer function" + ".npy"):
            TF = np.load("exc_" + str(i_inh) + "_" + str(i_adp) + "_" + "transfer function" + ".npy")*1e3
            diff2_fe_fe = np.load("exc_" + str(i_inh) + "_" + str(i_adp) + "_" + "second order derivation fe" + ".npy")
            diff2_fi_fi = np.load("exc_" + str(i_inh) + "_" + str(i_adp) + "_" + "second order derivation fi" + ".npy")
            diff2_fe_fi = np.load(
                "exc_" + str(i_inh) + "_" + str(i_adp) + "_" + "second order derivation fe and fi" + ".npy")

            fig, axs = plt.subplots(2, 2, figsize=(20, 20))
            fig.suptitle("excitatory derivation function components")

            max_limit = np.nanmax(TF) if np.nanmax(TF) < max_cut*1e3 else max_cut*1e3
            min_limit = np.nanmin(TF) if -np.nanmin(TF) < max_cut*1e3 else -max_cut*1e3
            if np.nanmin(TF) >= 0.0:
                im_1 = axs[0, 0].imshow(TF, norm=Normalize(vmin=min_limit, vmax=max_limit), cmap=cmap)
            else:
                linthresh = np.min([np.nanmax(TF), -np.nanmin(TF)]) * min_scale
                im_1 = axs[0, 0].imshow(TF, norm=SymLogNorm(linthresh=linthresh, vmin=min_limit, vmax=max_limit), cmap=cmap)
            fig.colorbar(im_1, ax=axs[0, 0])
            ticks_positions = np.array(np.around(np.linspace(0, inh_values.shape[0] - 1, 7)), dtype=int)
            axs[0, 0].set_yticks(ticks_positions)
            axs[0, 0].set_yticklabels(np.around(inh_values[ticks_positions], decimals=3) * 1e3)
            axs[0, 0].set_ylabel("inhibitory firing rate in Hz")
            ticks_positions = np.array(np.around(np.linspace(0, adaptation_values.shape[0] - 1, 7)), dtype=int)
            axs[0, 0].set_xticks(ticks_positions)
            axs[0, 0].set_xticklabels(np.around(adaptation_values[ticks_positions], decimals=0))
            axs[0, 0].set_xlabel("adaptation")
            axs[0, 0].set_title('Transfer function of inhibitory neurons in Hz')
            if np.nanmin(TF) < 0.0:
                axs[0, 0].contourf(np.arange(0, TF.shape[0], 1),
                                   np.arange(0, TF.shape[1], 1),
                                   TF[:, :, 0], levels=[-10000.0, 0.0],
                                   hatches=['/'], colors=['red'], alpha=0.0)
                axs[0, 0].contour(np.arange(0, TF.shape[0], 1),
                                  np.arange(0, TF.shape[1], 1),
                                  TF[:, :, 0], levels=[0.0], colors=['black'])

            max_limit = np.nanmax(diff2_fe_fe) if np.nanmax(diff2_fe_fe) < max_cut else max_cut
            min_limit = np.nanmin(diff2_fe_fe) if -np.nanmin(diff2_fe_fe) < max_cut else -max_cut
            linthresh = np.nanmin(diff2_fe_fe) if np.nanmin(diff2_fe_fe) > 0.0 else np.min([np.nanmax(diff2_fe_fe), -np.nanmin(diff2_fe_fe)]) * min_scale
            im_2 = axs[0, 1].imshow(diff2_fe_fe, norm=SymLogNorm(linthresh=linthresh, vmin=min_limit, vmax=max_limit), cmap=cmap)
            fig.colorbar(im_2, ax=axs[0, 1])
            axs[0, 1].set_title('$\partial^2f_e/\partial^2f$ TF')
            ticks_positions = np.array(np.around(np.linspace(0, inh_values.shape[0] - 1, 7)), dtype=int)
            axs[0, 1].set_yticks(ticks_positions)
            axs[0, 1].set_yticklabels(np.around(inh_values[ticks_positions], decimals=3) * 1e3)
            axs[0, 1].set_ylabel("inhibitory firing rate in Hz")
            ticks_positions = np.array(np.around(np.linspace(0, adaptation_values.shape[0] - 1, 7)), dtype=int)
            axs[0, 1].set_xticks(ticks_positions)
            axs[0, 1].set_xticklabels(np.around(adaptation_values[ticks_positions], decimals=0))
            axs[0, 1].set_xlabel("adaptation")
            if np.nanmin(diff2_fe_fe) < 0.0:
                axs[0, 1].contourf(np.arange(0, diff2_fe_fe.shape[0], 1),
                                   np.arange(0, diff2_fe_fe.shape[1], 1),
                                   diff2_fe_fe[:, :, 0], levels=[-10000.0, 0.0],
                                   hatches=['/'], colors=['red'], alpha=0.0)
                axs[0, 1].contour(np.arange(0, diff2_fe_fe.shape[0], 1),
                                  np.arange(0, diff2_fe_fe.shape[1], 1),
                                  diff2_fe_fe[:, :, 0], levels=[0.0], colors=['black'])

            max_limit = np.nanmax(diff2_fi_fi) if np.nanmax(diff2_fi_fi) < max_cut else max_cut
            min_limit = np.nanmin(diff2_fi_fi) if -np.nanmin(diff2_fi_fi) < max_cut else -max_cut
            linthresh = np.nanmin(diff2_fi_fi) if np.nanmin(diff2_fi_fi) > 0.0 else np.min([np.nanmax(diff2_fi_fi), -np.nanmin(diff2_fi_fi)]) * min_scale
            im_3 = axs[1, 0].imshow(diff2_fi_fi, norm=SymLogNorm(linthresh=linthresh, vmin=min_limit, vmax=max_limit), cmap=cmap)
            fig.colorbar(im_3, ax=axs[1, 0])
            axs[1, 0].set_title('$\partial^2f_i/\partial^2f$ TF')
            ticks_positions = np.array(np.around(np.linspace(0, inh_values.shape[0] - 1, 7)), dtype=int)
            axs[1, 0].set_yticks(ticks_positions)
            axs[1, 0].set_yticklabels(np.around(inh_values[ticks_positions], decimals=3) * 1e3)
            axs[1, 0].set_ylabel("inhibitory firing rate in Hz")
            ticks_positions = np.array(np.around(np.linspace(0, adaptation_values.shape[0] - 1, 7)), dtype=int)
            axs[1, 0].set_xticks(ticks_positions)
            axs[1, 0].set_xticklabels(np.around(adaptation_values[ticks_positions], decimals=0))
            axs[1, 0].set_xlabel("adaptation")
            if np.nanmin(diff2_fi_fi) < 0.0:
                axs[1, 0].contourf(np.arange(0, diff2_fi_fi.shape[0], 1),
                                   np.arange(0, diff2_fi_fi.shape[1], 1),
                                   diff2_fi_fi[:, :, 0], levels=[-10000.0, 0.0],
                                   hatches=['/'], colors=['red'], alpha=0.0)
                axs[1, 0].contour(np.arange(0, diff2_fi_fi.shape[0], 1),
                                  np.arange(0, diff2_fi_fi.shape[1], 1),
                                  diff2_fi_fi[:, :, 0], levels=[0.0], colors=['black'])

            max_limit = np.nanmax(diff2_fe_fi) if np.nanmax(diff2_fe_fi) < max_cut else max_cut
            min_limit = np.nanmin(diff2_fe_fi) if -np.nanmin(diff2_fe_fi) < max_cut else -max_cut
            linthresh = np.nanmin(diff2_fe_fi) if np.nanmin(diff2_fe_fi) > 0.0 else np.min([np.nanmax(diff2_fe_fi), -np.nanmin(diff2_fe_fi)]) * min_scale
            im_4 = axs[1, 1].imshow(diff2_fe_fi, norm=SymLogNorm(linthresh=linthresh, vmin=min_limit, vmax=max_limit), cmap=cmap)
            fig.colorbar(im_4, ax=axs[1, 1])
            ticks_positions = np.array(np.around(np.linspace(0, inh_values.shape[0] - 1, 7)), dtype=int)
            axs[1, 1].set_yticks(ticks_positions)
            axs[1, 1].set_yticklabels(np.around(inh_values[ticks_positions], decimals=3) * 1e3)
            axs[1, 1].set_ylabel("inhibitory firing rate in Hz")
            ticks_positions = np.array(np.around(np.linspace(0, adaptation_values.shape[0] - 1, 7)), dtype=int)
            axs[1, 1].set_xticks(ticks_positions)
            axs[1, 1].set_xticklabels(np.around(adaptation_values[ticks_positions], decimals=0))
            axs[1, 1].set_xlabel("adaptation")
            axs[1, 1].set_title('$\partial f_i\partial f_e/\partial^2f$ TF')
            if np.nanmin(diff2_fe_fi) < 0.0:
                axs[1, 1].contourf(np.arange(0, diff2_fe_fi.shape[0], 1),
                                   np.arange(0, diff2_fe_fi.shape[1], 1),
                                   diff2_fe_fi[:, :, 0], levels=[-10000.0, 0.0],
                                   hatches=['/'], colors=['red'], alpha=0.0)
                axs[1, 1].contour(np.arange(0, diff2_fe_fi.shape[0], 1),
                                  np.arange(0, diff2_fe_fi.shape[1], 1),
                                  diff2_fe_fi[:, :, 0], levels=[0.0], colors=['black'])
        plt.savefig("exc_" + str(i_inh) + "_" + str(i_adp)+'.svg')

plt.show()

# coupling = np.array([0.0]).reshape((1, 1))
#
# for inh_values in [np.linspace(0.0, 200.0, 100) * 1e-3, np.linspace(0.0, 10.0, 100) * 1e-3]:
#     adaptation_values = np.linspace(0.0, 100, 1000)
#     result_max = np.empty((inh_values.shape[0], adaptation_values.shape[0], 7))
#     result_mid = np.empty((inh_values.shape[0], adaptation_values.shape[0], 7))
#     result_mid_2 = np.empty((inh_values.shape[0], adaptation_values.shape[0], 7))
#     result_min = np.empty((inh_values.shape[0], adaptation_values.shape[0], 7))
#     for index_inh, inh in enumerate(inh_values):
#         for index_adpt, adpt in enumerate(adaptation_values):
#             input_max = np.array([-1e-6, inh, 1.0, 1.0, 1.0, adpt, 0.0]).reshape((7, 1))
#             result_max[index_inh, index_adpt, :] = Zerlaut_fit.dfun(input_max, coupling).squeeze(1)
#             input_mid = np.array([-1e-6, inh, 0.0, 0.0, 0.0, adpt, 0.0]).reshape((7, 1))
#             result_mid[index_inh, index_adpt, :] = Zerlaut_fit.dfun(input_mid, coupling).squeeze(1)
#             input_mid_2 = np.array([-1e-6, inh, 0.0, 0.0, 1.0, adpt, 0.0]).reshape((7, 1))
#             result_mid_2[index_inh, index_adpt, :] = Zerlaut_fit.dfun(input_mid_2, coupling).squeeze(1)
#             input_min = np.array([-1e-6, inh, 0.0, -1.0, 0.0, adpt, 0.0]).reshape((7, 1))
#             result_min[index_inh, index_adpt, :] = Zerlaut_fit.dfun(input_min, coupling).squeeze(1)
#             # print(result_max[index_inh, index_adpt, 0], result_min[index_inh, index_adpt, 0], end=' ')
#             print("\r index :", index_adpt, index_inh, end='');
#             sys.stdout.flush()
#     fig = plt.figure()
#     ax = fig.add_subplot(projection='3d')
#     X, Y = np.meshgrid(adaptation_values ,inh_values)
#     Z = result_max[:, :, 0]
#     ax.plot_wireframe(X, Y, Z)
#     plt.title('ex max')
#
#     fig = plt.figure()
#     ax = fig.add_subplot(projection='3d')
#     X, Y = np.meshgrid(adaptation_values ,inh_values)
#     Z = result_mid[:, :, 0]
#     ax.plot_wireframe(X, Y, Z)
#     plt.title('ex mid')
#
#     fig = plt.figure()
#     ax = fig.add_subplot(projection='3d')
#     X, Y = np.meshgrid(adaptation_values ,inh_values)
#     Z = result_mid_2[:, :, 0]
#     ax.plot_wireframe(X, Y, Z)
#     plt.title('ex mid')
#
#     fig = plt.figure()
#     ax = fig.add_subplot(projection='3d')
#     X, Y = np.meshgrid(adaptation_values ,inh_values)
#     Z = result_min[:, :, 0]
#     ax.plot_wireframe(X, Y, Z)
#     plt.title('ex min')
#
# ex_values = np.concatenate([[-1e-6], np.linspace(0.0, 200.0, 100) * 1e-3])
# adaptation_values = np.linspace(0.0, 100, 1000)
# result_max_in = np.empty((ex_values.shape[0], adaptation_values.shape[0], 7))
# result_mid_in = np.empty((ex_values.shape[0], adaptation_values.shape[0], 7))
# result_mid_2_in = np.empty((ex_values.shape[0], adaptation_values.shape[0], 7))
# result_min_in = np.empty((ex_values.shape[0], adaptation_values.shape[0], 7))
# for index_ex, ex in enumerate(ex_values):
#     for index_adpt, adpt in enumerate(adaptation_values):
#         input_max = np.array([ex, 0.0, 1.0, 1.0, 1.0, adpt, 0.0]).reshape((7, 1))
#         result_max_in[index_ex, index_adpt, :] = Zerlaut_fit.dfun(input_max, coupling).squeeze(1)
#         input_mid = np.array([ex, 0.0, 0.0, 0.0, 0.0, adpt, 0.0]).reshape((7, 1))
#         result_mid_in[index_ex, index_adpt, :] = Zerlaut_fit.dfun(input_mid, coupling).squeeze(1)
#         input_mid_2 = np.array([ex, 1.0, 0.0, 0.0, 0.0, adpt, 0.0]).reshape((7, 1))
#         result_mid_2_in[index_ex, index_adpt, :] = Zerlaut_fit.dfun(input_mid_2, coupling).squeeze(1)
#         input_min = np.array([ex, 0.0, 0.0, -1.0, 0.0, adpt, 0.0]).reshape((7, 1))
#         result_min_in[index_ex, index_adpt, :] = Zerlaut_fit.dfun(input_min, coupling).squeeze(1)
#         print("\r index :", index_adpt, index_ex, end='');
#         sys.stdout.flush()
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# X, Y = np.meshgrid(adaptation_values, ex_values)
# Z = result_max_in[:, :, 1]
# ax.plot_wireframe(X, Y, Z)
# plt.title('in max')
#
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# X, Y = np.meshgrid(adaptation_values, ex_values)
# Z = result_mid_in[:, :, 1]
# ax.plot_wireframe(X, Y, Z)
# plt.title('in mid')
#
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# X, Y = np.meshgrid(adaptation_values, ex_values)
# Z = result_mid_2_in[:, :, 1]
# ax.plot_wireframe(X, Y, Z)
# plt.title('in mid')
#
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# X, Y = np.meshgrid(adaptation_values, ex_values)
# Z = result_min_in[:, :, 1]
# ax.plot_wireframe(X, Y, Z)
# plt.title('in min')
# plt.show()
