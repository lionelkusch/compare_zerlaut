import numpy as np
import scipy.special as sp_spec
from scipy.optimize import minimize
from nest_elephant_tvb.Tvb.modify_tvb.Zerlaut import ZerlautAdaptationSecondOrder as model

def create_transfer_function(parameter,excitatory):
    """
    create the transfer function from the model of Zerlaut adapted for inhibitory and excitatory neurons
    :param parameter: parameter of the simulation
    :param excitatory: if for excitatory or inhibitory population
    :return: the transfer function
    """
    model_test = model()
    model_test.g_L = np.array(parameter['g_L'])
    model_test.E_L_e =  np.array(parameter['E_L'])
    model_test.E_L_i = np.array(parameter['E_L'])
    model_test.C_m = np.array(parameter['C_m'])
    model_test.b_e = np.array(parameter['b'])
    model_test.a_e = np.array(parameter['a'])
    model_test.b_i = np.array(parameter['b'])
    model_test.a_i = np.array(parameter['a'])
    model_test.tau_w_e = np.array(parameter['tau_w'])
    model_test.tau_w_i = np.array(parameter['tau_w'])
    model_test.E_e = np.array(parameter['E_ex'])
    model_test.E_i = np.array(parameter['E_in'])
    model_test.Q_e = np.array(parameter['Q_e'])
    model_test.Q_i = np.array(parameter['Q_i'])
    model_test.tau_e = np.array(parameter['tau_syn_ex'])
    model_test.tau_i = np.array(parameter['tau_syn_in'])
    model_test.N_tot = np.array(parameter['N_tot'])
    model_test.p_connect_e = np.array(parameter['p_connect_ex'])
    model_test.p_connect_i = np.array(parameter['p_connect_ex'])
    model_test.g = np.array(parameter['g'])
    model_test.T = np.array(parameter['t_ref'])
    model_test.external_input_in_in = np.array(0.0)
    model_test.external_input_in_ex = np.array(0.0)
    model_test.external_input_ex_in = np.array(0.0)
    model_test.external_input_ex_ex = np.array(0.0)
    model_test.K_ext_e=np.array(1)
    model_test.K_ext_i=np.array(0)
    if excitatory:
        def TF(fe,fi,p,f_ext_e=0.0,f_ext_i=0.0,w=0.0):
            model_test.P_e=p
            return model_test.TF_excitatory(fe,fi,f_ext_e,f_ext_i,w)
    else:
        def TF(fe,fi,p,f_ext_e=0.0,f_ext_i=0.0,w=0.0):
            model_test.P_i=p
            return model_test.TF_inhibitory(fe,fi,f_ext_e,f_ext_i,w)
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
    Vthre_eff = muV+np.sqrt(2)*sV*sp_spec.erfcinv(rate*2.*Tv) # effective threshold
    return Vthre_eff

def fitting_1(feOut,feSim, fiSim, adaptation, parameters, nb_value_fexc,nb_value_finh,nb_value_adaptation,
              MINadaptation, MAXadaptation,MINfinh,MAXfinh,MAXfexc,excitatory):

    #Compute mean of value for the model
    muV, sV, Tv =model.get_fluct_regime_vars(feSim,fiSim,0.00,0.0,adaptation,parameters['Q_e'],parameters['tau_syn_ex'],parameters['E_ex'],parameters['Q_i'],parameters['tau_syn_in'],parameters['E_in'],
                                             parameters['g_L'],parameters['C_m'],parameters['E_L'],parameters['N_tot'],parameters['p_connect_ex'],parameters['p_connect_in'],parameters['g'],0.0,0.0)
    Tv+= parameters['g_L']/parameters['C_m']
    i_non_zeros = np.where(feOut*Tv<1.0)
    Vthre_eff = effective_Vthre(feOut[i_non_zeros], muV[i_non_zeros], sV[i_non_zeros], Tv[i_non_zeros])*1e-3
    TvN = Tv[i_non_zeros]*parameters['g_L']/parameters['C_m']

    # initialisation of the fitting
    TF = create_transfer_function(parameters,excitatory=excitatory)
    P = np.zeros(10)
    P[:5] = Vthre_eff.mean(), 1e-3, 1e-3, 1e-3, 1e-3

    # fitting the voltage threshold
    def Res(p):
        pp=p
        vthre = model.threshold_func(muV[i_non_zeros], sV[i_non_zeros], TvN,  *pp)
        return np.mean((Vthre_eff-vthre)**2)
    plsq = minimize(Res, P, method='SLSQP',options={
        'ftol': 1e-15,
        'disp': True,
        'maxiter':50000})

    # fitting the mean firing rate
    P = plsq.x
    print("error ", np.mean(((feOut - TF(feSim, fiSim, P,w=adaptation)) ** 2) * 1e3))
    def Res_2(p):
        '''
        absolute error
        :param p: polynome
        :return:
        '''
        return np.mean(((feOut - TF(feSim,fiSim,p,w=adaptation))*1e3) ** 2)

    def Res_2_bis(p):
        """
        relative error
        :param p: polynomial
        :return:
        """
        return np.mean(((feOut - TF(feSim, fiSim, p, w=adaptation)) /feOut) ** 2)

    print("with adaptation absolute error")
    plsq = minimize(Res_2, P, method='nelder-mead',
                    tol=1e-11,
                    # tol=1e-7,
                    options={
                        'xatol': 1e-15,
                        'disp': True,
                        'maxiter': 50000})
    P = plsq.x
    print("with adaptation relative error")
    plsq_2 = minimize(Res_2_bis, P, method='nelder-mead',
                      tol=1e-11,
                      # tol=1e-7,
                      options={
                          'xatol': 1e-15,
                          'disp': True,
                          'maxiter': 50000})
    P_2 = plsq_2.x
    p_with = P

    # without adaptation
    mask = np.where(adaptation == 0.0)
    feOut_1 = feOut[mask]
    feSim_1 = feSim[mask]
    fiSim_1 = fiSim[mask]
    adaptation_1 = adaptation[mask]
    #Compute mean of value for the model
    muV, sV, Tv =model.get_fluct_regime_vars(feSim_1,fiSim_1,0.00,0.0,adaptation_1,parameters['Q_e'],parameters['tau_syn_ex'],parameters['E_ex'],parameters['Q_i'],parameters['tau_syn_in'],parameters['E_in'],
                                             parameters['g_L'],parameters['C_m'],parameters['E_L'],parameters['N_tot'],parameters['p_connect_ex'], parameters['p_connect_in'],parameters['g'],0.0,0.0)
    Tv+= parameters['g_L']/parameters['C_m']
    i_non_zeros = np.where(feOut_1*Tv<1.0)
    Vthre_eff = effective_Vthre(feOut_1[i_non_zeros], muV[i_non_zeros], sV[i_non_zeros], Tv[i_non_zeros])*1e-3
    TvN = Tv[i_non_zeros]*parameters['g_L']/parameters['C_m']

    # initialisation of the fitting
    TF = create_transfer_function(parameters,excitatory=excitatory)
    P = np.zeros(10)
    P[:5] = Vthre_eff.mean(), 1e-3, 1e-3, 1e-3, 1e-3

    # fitting the voltage threshold
    def Res(p):
        pp=p
        vthre = model.threshold_func(muV[i_non_zeros], sV[i_non_zeros], TvN, *pp)
        return np.mean((Vthre_eff-vthre)**2)
    plsq = minimize(Res, P, method='SLSQP',options={
        'ftol': 1e-15,
        'disp': True,
        'maxiter':40000})

    P = plsq.x
    print("error ", np.mean(((feOut - TF(feSim, fiSim, P,w=adaptation)) * 1e3) ** 2))
    def Res_2(p):
        """
        absolute error
        :param p: polynomial
        :param p:
        :return:
        """
        return np.mean(((feOut_1 - TF(feSim_1,fiSim_1,p,w=adaptation_1)) *1e3)** 2)
    def Res_2_bis(p):
        """
        relative error
        :param p: polynomial
        :return:
        """
        return np.mean(((feOut_1 - TF(feSim_1, fiSim_1, p, w=adaptation_1)) / feOut_1) ** 2)
    print("no adaptation absolute error")
    plsq = minimize(Res_2, P, method='nelder-mead',
                    tol=1e-11,
                    # tol=1e-7,
                    options={
                        'xtol': 1e-15,
                        'disp': True,
                        'maxiter': 50000})
    P = plsq.x
    print("no adaptation relative error")
    plsq_2 = minimize(Res_2_bis, P, method='nelder-mead',
                      tol=1e-11,
                      # tol=1e-7,
                      options={
                          'xtol': 1e-15,
                          'disp': True,
                          'maxiter': 50000})
    P_2 = plsq_2.x


    p_without = P

    np.set_printoptions(edgeitems=3,infstr='inf', linewidth=75, nanstr='nan', precision=8, suppress=False, threshold=1000, formatter=None)

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
    index = np.argsort(np.abs(((feOut - TF(feSim, fiSim, p_without,w=adaptation)) * 1e3)))[-5:]
    print("frequency ex", feSim[index]*1e3)
    print("frequency in", fiSim[index]*1e3)
    print("adaptation", adaptation[index])
    print("expected : ", feOut[index]*1e3)
    print("got : ", TF(feSim, fiSim, p_without,w=adaptation)[index] * 1e3)
    print("error : ", np.abs(((feOut - TF(feSim,fiSim, p_without,w=adaptation)) * 1e3))[index])
    print("max error ", np.abs(((feOut - TF(feSim, fiSim, p_without,w=adaptation)) * 1e3))[index[-1]])
    print("error ", np.mean(((feOut - TF(feSim, fiSim, p_without,w=adaptation)) * 1e3) ** 2))
    print("error relative ", np.mean(((feOut - TF(feSim, fiSim, p_without,w=adaptation)) / feOut) ** 2))
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
    index = np.argsort(np.abs(((feOut - TF(feSim, fiSim, p_with,w=adaptation)) * 1e3)))[-5:]
    print("frequency ex", feSim[index]*1e3)
    print("frequency in", fiSim[index]*1e3)
    print("adaptation", adaptation[index])
    print("expected : ", feOut[index]*1e3)
    print("got : ", TF(feSim, fiSim, p_with,w=adaptation)[index] * 1e3)
    print("error : ", np.abs(((feOut - TF(feSim,fiSim, p_with,w=adaptation)) * 1e3))[index])
    print("max error ", np.abs(((feOut - TF(feSim, fiSim, p_with,w=adaptation)) * 1e3))[index[-1]])
    print("error ", np.mean(((feOut - TF(feSim, fiSim, p_with,w=adaptation)) * 1e3)** 2 ))
    print("error relative ", np.mean(((feOut - TF(feSim, fiSim, p_with,w=adaptation)) /feOut)** 2 ))
    print("##########################################################################")
    print(p_with)

    return p_with,p_without,TF


    # Function excitatory
# ######################## fitting without adaptation ######################
#                     #### data without adaptation    #####
# frequency ex [0.6389375  0.61120833 0.55575    1.01328125 0.58347917]
# frequency in [0. 0. 0. 0. 0.]
# adaptation [0. 0. 0. 0. 0.]
# expected :  [ 2.69   1.888  0.678 19.858  1.16 ]
# got :  [ 3.41175563  2.65647053  1.45593215 20.66695693  2.00401546]
# error :  [0.72175563 0.76847053 0.77793215 0.80895693 0.84401546]
# max error  0.8440154572877832
# error  0.04039305320842687
# error relative  0.008161469274485652
#                     #### all data                   ####
# frequency ex [2.10583333 2.03020833 2.05541667 1.97979167 2.13104167]
# frequency in [2.10526316 2.10526316 2.10526316 2.10526316 2.10526316]
# adaptation [100. 100. 100. 100. 100.]
# expected :  [11.976  9.924 10.566  8.588 12.552]
# got :  [14.91699689 12.89968488 13.56578254 11.5911157  15.60037577]
# error :  [2.94099689 2.97568488 2.99978254 3.0031157  3.04837577]
# max error  3.048375774875902
# error  0.7030041374646141
# error relative  0.09088932349774831
# ##########################################################################
# [-0.05640304  0.00730611 -0.01454463  0.01104506 -0.00034363 -0.00256904
#  -0.03369905  0.00804323 -0.00276914 -0.02988127]
# ######################## fitting with adaptation    ######################
#                     #### data without adaptation    #####
# frequency ex [1.75291667 1.7025     1.65208333 1.72770833 1.67729167]
# frequency in [2.10526316 2.10526316 2.10526316 2.10526316 2.10526316]
# adaptation [0. 0. 0. 0. 0.]
# expected :  [15.898 14.374 12.95  15.152 13.672]
# got :  [13.89681811 12.36589658 10.91997966 13.12102746 11.63202578]
# error :  [2.00118189 2.00810342 2.03002034 2.03097254 2.03997422]
# max error  2.039974224858276
# error  0.4964011875307476
# error relative  0.009472964103212266
#                     #### all data                   ####
# frequency ex [1.72770833 1.4680625  1.67729167 0.99666667 1.49579167]
# frequency in [2.10526316 0.         2.10526316 0.         0.        ]
# adaptation [  0.          94.73684211   0.         100.         100.        ]
# expected :  [15.152 19.668 13.672  1.066 19.696]
# got :  [13.12102746 21.70571714 11.63202578  3.12653964 21.95185582]
# error :  [2.03097254 2.03771714 2.03997422 2.06053964 2.25585582]
# max error  2.2558558242497457
# error  0.23534237193785973
# error relative  0.06740949887038498
# ##########################################################################
# [-5.55766279e-02  4.96819240e-03 -4.37691932e-03  1.19052623e-02
#  -8.98669268e-04  1.13529702e-03 -2.16432326e-02  1.29734790e-05
#   1.91164145e-03 -5.40666443e-03]


# Function inhibitory
# ######################## fitting without adaptation ######################
#                     #### data without adaptation    #####
# frequency ex [0.540625   1.0975     0.83052083 0.51541667 0.843125  ]
# frequency in [0.         2.10526316 0.         0.         0.        ]
# adaptation [0. 0. 0. 0. 0.]
# expected :  [ 2.396  3.686 18.48   1.602 19.232]
# got :  [ 2.859458    4.15895251 18.97505144  2.14872388 19.89190839]
# error :  [0.463458   0.47295251 0.49505144 0.54672388 0.65990839]
# max error  0.659908392056055
# error  0.02771532003592755
# error relative  0.0045226572264743465
#                     #### all data                   ####
# frequency ex [1.29916667 1.29916667 1.31177083 2.07936458 1.324375  ]
# frequency in [0.         0.         0.         2.10526316 0.        ]
# adaptation [100.          94.73684211 100.         100.         100.        ]
# expected :  [18.424 19.734 19.14  19.626 19.76 ]
# got :  [21.8960201  23.24316768 22.6539652  23.14245145 23.41771778]
# error :  [3.4720201  3.50916768 3.5139652  3.51645145 3.65771778]
# max error  3.6577177774473233
# error  0.8245368515247785
# error relative  0.04764457182566207
# ##########################################################################
# [-5.69379217e-02  5.05332087e-03 -4.20746608e-03  1.10093910e-02
#  -1.31602263e-04 -6.89239312e-04 -1.70501058e-02  2.24002256e-03
#  -8.41613042e-05 -1.29776194e-02]
# ######################## fitting with adaptation    ######################
#                     #### data without adaptation    #####
# frequency ex [1.60166667 1.5890625  1.52604167 1.56385417 1.57645833]
# frequency in [2.10526316 2.10526316 2.10526316 2.10526316 2.10526316]
# adaptation [0. 0. 0. 0. 0.]
# expected :  [19.644 19.202 16.986 18.32  18.928]
# got :  [17.30766047 16.84501992 14.59900593 15.93275107 16.38666937]
# error :  [2.33633953 2.35698008 2.38699407 2.38724893 2.54133063]
# max error  2.5413306330103818
# error  0.5178384182451361
# error relative  0.008034120433518647
#                     #### all data                   ####
# frequency ex [1.60166667 1.5890625  1.52604167 1.56385417 1.57645833]
# frequency in [2.10526316 2.10526316 2.10526316 2.10526316 2.10526316]
# adaptation [0. 0. 0. 0. 0.]
# expected :  [19.644 19.202 16.986 18.32  18.928]
# got :  [17.30766047 16.84501992 14.59900593 15.93275107 16.38666937]
# error :  [2.33633953 2.35698008 2.38699407 2.38724893 2.54133063]
# max error  2.5413306330103818
# error  0.22129510442305217
# error relative  0.02181710431354046
# ##########################################################################
# [-0.05657417  0.00435497 -0.0019183   0.0127786  -0.00037256 -0.00101983
#  -0.01505324  0.00044806  0.00152825 -0.00372715]

