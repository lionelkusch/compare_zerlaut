import matplotlib.pyplot as plt
import os
import numpy as np
# from nest_elephant_tvb.simulation.file_tvb.Zerlaut import ZerlautAdaptationSecondOrder as model
from tvb.simulator.integrators import HeunDeterministic
test1=False
from nest_elephant_tvb.Tvb.modify_tvb.Zerlaut import ZerlautAdaptationFirstOrder as model
# test1=True
# from nest_elephant_tvb.simulation.file_tvb.Zerlaut_test_1 import ZerlautAdaptationFirstOrder as model

# excitatory parameter should came from the parameter file
excitatory={
        'C_m':200.0,
        't_ref':5.0,
        'V_reset':-64.5,
        'E_L':-64.5,
        'g_L':10.0,
        'I_e':0.0,
        'a':0.0,
        'b':0.0,
        'Delta_T':2.0,
        'tau_w':500.0,
        'V_th':-50.0,
        'E_ex':0.0,
        'tau_syn_ex':5.0,
        'E_in':-80.0,
        'tau_syn_in':5.0,
    'V_peak': 10.0,
    'N_tot':10**4,
    'p_connect':0.05,
    'g':0.2,
    'Q_e':1.0,
    'Q_i':3.5,
}
#inhibitory
inhibitory={
        'C_m':200.0,
        't_ref':5.0,
        'V_reset':-65.0,
        'E_L':-65.,
        'g_L':10.0,
        'I_e':0.0,
        'a':0.0,
        'b':0.0,
        'Delta_T':0.5,
        'tau_w':1.0,
        'V_th':-50.0,
        'E_ex':0.0,
        'tau_syn_ex':5.0,
        'E_in':-80.0,
        'tau_syn_in':5.0,
    'V_peak': 10.0,
    'N_tot':10**4,
    'p_connect':0.05,
    'g':0.2,
    'Q_e':1.0,
    'Q_i':3.5
}


def compute_rate(data,begin,end,nb):
    """
    Compute the firing rate
    :param data: the spike of all neurons between end and begin
    :param begin: the time of the first spike
    :param end: the time of the last spike
    :return: the mean firing rate
    """
    #get data
    n_fil = data[:, 0]
    n_fil = n_fil.astype(int)
    #count the number of the same id
    count_of_n = np.bincount(n_fil)
    #compute the rate
    rate_each_n_incomplet = count_of_n / (end - begin)
    #fill the table with the neurons which are not firing
    rate_each_n = np.concatenate(
        (rate_each_n_incomplet, np.zeros(-np.shape(rate_each_n_incomplet)[0] + nb +1)))
    #save the value
    return rate_each_n[1:]

def load_event(events):
    """
    Get the id of the neurons which create the spike and time
    :param events: id and time of each spikes
    :return: The spike of all neurons
    """
    data_concatenated =  np.concatenate(([events['senders']],[events['times']]))
    if data_concatenated.size < 5:
        print('empty file')
        return None
    data_raw = data_concatenated[np.argsort(data_concatenated[:, 1])]
    return np.swapaxes(data_raw,0,1)

def load_spike(path):
    """
    Get the id of the neurons which create the spike and time
    :param path: the path to the file
    :return: The spike of all neurons
    """
    if not os.path.exists(path + "/spike_detector.gdf"):
        print('no file')
        return None
    data_concatenated = np.loadtxt(path + "/spike_detector.gdf")
    if data_concatenated.size < 5:
        print('empty file')
        return None
    data_raw = data_concatenated[np.argsort(data_concatenated[:, 1])]
    return data_raw

def create_model_integration(parameter_ex,parameter_in,k,test1):
    '''
    create the mean field model from the parameters
    :param parameter_ex: parameters for excitatory neurons
    :param parameter_in:  parameters for inhibitory neurons
    :param k: number of external connections
    :return: function for fitting polynome
    '''
    model_test = model()
    model_test.g_L = np.array(parameter_ex['g_L'])
    model_test.E_L_e =  np.array(parameter_ex['E_L'])
    model_test.E_L_i = np.array(parameter_in['E_L'])
    model_test.C_m = np.array(parameter_ex['C_m'])
    model_test.b_e = np.array(parameter_ex['b'])
    model_test.a_e = np.array(parameter_ex['a'])
    model_test.b_i = np.array(parameter_in['b'])
    model_test.a_i = np.array(parameter_in['a'])
    model_test.tau_w_e = np.array(parameter_ex['tau_w'])
    model_test.tau_w_i = np.array(parameter_in['tau_w'])
    model_test.E_e = np.array(parameter_ex['E_ex'])
    model_test.E_i = np.array(parameter_ex['E_in'])
    model_test.Q_e = np.array(parameter_ex['Q_e'])
    model_test.Q_i = np.array(parameter_ex['Q_i'])
    model_test.tau_e = np.array(parameter_ex['tau_syn_ex'])
    model_test.tau_i = np.array(parameter_ex['tau_syn_in'])
    model_test.N_tot = np.array(parameter_ex['N_tot'])
    model_test.p_connect = np.array(parameter_ex['p_connect'])
    model_test.g = np.array(parameter_ex['g'])
    model_test.T = np.array(parameter_ex['t_ref'])
    model_test.external_input_in_in = np.array(0.0)
    model_test.external_input_in_ex = np.array(0.0)
    model_test.external_input_ex_in = np.array(0.0)
    model_test.external_input_ex_ex = np.array(0.0)
    model_test.K_ext_e=np.array(k)
    model_test.K_ext_i=np.array(0)
    integrator = HeunDeterministic(dt=0.1)
    integrator.configure()
    if test1:
        def function(p,fe,fi,f_ext,w):
            model_test.P_e = p[:15]
            model_test.P_i = p[15:]
            x = np.concatenate([fe,fi,w,np.zeros((fe.size))]).reshape((4,fe.size,1,1))
            coupling = np.array([f_ext]).reshape((1,f_ext.size,1,1))
            local_coupling=np.array([0.])
            stimulus=np.array([0.])
            return integrator.scheme(x,model_test.dfun,coupling,local_coupling,stimulus)
    else:
        def function(p,fe,fi,f_ext,w):
            model_test.P_e = p[:10]
            model_test.P_i = p[10:]
            x = np.concatenate([fe,fi,w,np.zeros((fe.size))]).reshape((4,fe.size,1,1))
            coupling = np.array([f_ext]).reshape((1,f_ext.size,1,1))
            local_coupling=np.array([0.])
            stimulus=np.array([0.])
            return integrator.scheme(x,model_test.dfun,coupling,local_coupling,stimulus)
    return function

def run_test (n,p,fe,fi,f_ext,adaptation,function):
    """
    function one simulation with the data
    :param n: number of step
    :param p: polynome of the transfer function
    :param fe: mean firing rate of excitatory population
    :param fi: mean firing rate of inhibitory population
    :param f_ext: mean firing rate of external excitatory population
    :param adaptation: the adaptation of the excitatory population
    :param function: the function for  the simulation
    :return: the result of the simulation
    """
    sim = np.empty((4,fe.size))
    sim[:,0]=[fe[0],fi[0],adaptation[0],0.0]
    for i in range(1,n):
        if i%1000 == 0:
            print(i)
        sim[:,i] = function(p,np.array([sim[0,i-1]]),np.array([sim[1,i-1]]),np.array([f_ext[i-1]]),np.array([sim[2,i-1]]))[:,0,0,0]
    return sim

def run_fit (n,p,fe,fi,f_ext,adaptation,function):
    """
    function one simulation with only initialisation
    :param n: number of step
    :param p: polynome of the transfer function
    :param fe: mean firing rate of excitatory population init
    :param fi: mean firing rate of inhibitory population init
    :param f_ext: mean firing rate of external excitatory population
    :param adaptation: the adaptation of the excitatory population
    :param function: the function for  the simulation
    :return: the result of the simulation
    """
    sim = np.empty((4,fe.size,n))
    sim[:,:,0]=[fe,fi,adaptation,np.repeat(0.0,fe.size)]
    for i in range(1,n):
        sim[:,:,i] = function(p,np.array(sim[0,:,i-1]),np.array(sim[1,:,i-1]),np.array([f_ext[i:-n+i]]),np.array(sim[2,:,i-1]))[:,:,0,0]
    return sim


def engin(parameters_ex,parameters_in,k,name_file,test1=False):
    """
    run the fitting
    :param parameters_ex:
    :param parameters_in:
    :param k:
    :param name_file:
    :return:
    """
    parameters_ex['b']=6
    data = np.load(name_file, encoding = 'latin1',allow_pickle=True)
    shift_time = 0
    max_time = shift_time+20000
    nb_step=100
    # max_time = 29801
    fe = data[2][shift_time:max_time]*1e-3
    fi = data[1][shift_time:max_time]*1e-3
    f_ext = data[3][shift_time:max_time]*1e-3
    shift = 198
    adapation = data[-1][shift+shift_time:max_time+shift]

    expected = np.array([fe[nb_step:],fi[nb_step:],adapation[nb_step:]])

    function = create_model_integration(parameters_ex,parameters_in,k,test1)
    # if test1:
    #     P = np.ones(30)*0.
    #     P[0]=-0.06
    #     P[15]=-0.065
    # else:
    #     P = np.ones(20)*0.
    #     P[0]=-0.06
    #     P[10]=-0.065
    # def Res(p):
    #     # P_e = p[:10]
    #     # P_i = p[10:]
    #     # def Res_e(p):
    #     #     res =run_fit(nb_step,np.concatenate((p,P_i)),fe[:-nb_step],fi[:-nb_step],f_ext,adapation[:-nb_step],function)
    #     #     fe_error =  (res[0,:,nb_step-1]-expected[0,:])
    #     #     return np.mean(np.abs(fe_error))*1e3
    #     # plsq = minimize(Res_e, P_e, method='nelder-mead',tol=1e-12, \
    #     #             options={'xtol': 0.01, 'disp': True, 'maxiter': 100000, 'maxfev':100000})
    #     # P_e = plsq.x
    #     # # print(P_e)
    #     # def Res_i(p):
    #     #     res =run_fit(nb_step,np.concatenate((P_e,p)),fe[:-nb_step],fi[:-nb_step],f_ext,adapation[:-nb_step],function)
    #     #     fi_error =  (res[0,:,nb_step-1]-expected[0,:])
    #     #     return np.mean(np.abs(fi_error))
    #     # plsq = minimize(Res_i, P_i, method='nelder-mead',tol=1e-12, \
    #     #                 options={'xtol': 0.01, 'disp': True, 'maxiter': 100000, 'maxfev':100000})
    #     # print(P_i)
    #     # P = np.concatenate((P_e,plsq.x))
    #     def Res_all(P):
    #         res =run_fit(nb_step,P,fe[:-nb_step],fi[:-nb_step],f_ext,adapation[:-nb_step],function)
    #         f_error =  np.concatenate(((res[1,:,nb_step-1]-expected[1,:])*1e3,(res[0,:,nb_step-1]-expected[0,:])*1e3))
    #         return np.mean(np.abs(f_error))
    #     plsq = minimize(Res_all, P, method='nelder-mead',tol=1e-12, \
    #                     options={'xtol': 0.01, 'disp': True, 'maxiter': 100000, 'maxfev':100000})
    #     # print(plsq.x)
    #     return plsq.x
    #
    # def test_result (p):
    #     return run_test(fe.size,p,fe,fi,adapation,f_ext,function)
    # res= test_result(P)
    # error = np.sum(np.abs(res[2,:]-adapation))
    # error=10.0
    # print(error)
    # step = 0
    # while error > 0.1 and step < 5:
    #     P=Res(P)
    #     res= test_result(P)
    #     error = np.sum(np.abs(res[2,:]-adapation))
    #     print(error)
    #     print(P[:10])
    #     print(P[10:])
    #     step+=1

    # P=np.array(
        # [-0.04934242,  0.00143687,  0.01500461, -0.00057953,  0.00337665, -0.00213777,  0.00048221, -0.00708383,  0.00436972, -0.00745851, -0.04850237, -0.00289845,  0.00208314, -0.01194692,  0.00573622, -0.0612846,  0.01000852,  0.01638475,  0.01848123, -0.03334372]
        # [-0.05059317,  0.0036078,   0.01794401,  0.00467008,  0.00098553,  0.0082953, -0.00985289, -0.02600252, -0.00274499, -0.01051463, -0.05084858,  0.00146866, -0.00657981,  0.0014993,  -0.0003816,   0.00020026,  0.00078719, -0.00322428, -0.00842626, -0.0012793 ]
    # )

    P = np.array([-0.05186354, 0.00311893, -0.02345989, -0.00254818, 0.00089175, -0.00387891, -0.02513742, 0.00877677,
         0.00181444, -0.01943301, -5.27027092e-02, 3.13101743e-03, -1.61010481e-02, 3.73668144e-05, 6.57077594e-04,
         -6.18163939e-03, -2.07741770e-02, 6.18305441e-03, 1.50645268e-03, -1.98705312e-02])
    # P=np.array([-0.04861008, -0.00032855, -0.00160813, -0.00597536,  0.00189784,  0.00740554, -0.00069605,  0.00209382,  0.00364572,  0.00216413 ,-0.05039001,  0.0006127 , -0.00019834, -0.00155777,  0.00141241,   0.0025035 , -0.00376844,  0.00073833,  0.00223948, -0.00351208])
    res = function(P,fe[:-nb_step],fi[:-nb_step],f_ext[:-nb_step],adapation[:-nb_step])[:,:,0,0]
    plt.figure()
    plt.plot(fe*1e3,label="excitatory")
    plt.plot(fi*1e3,label="inhibitory")
    plt.plot(adapation,label="adaptation excitatory")
    plt.plot(f_ext*1e3/(300/k),label='external input')
    plt.legend()
    plt.figure()
    plt.plot(res[0,:]*1e3,label="excitatory")
    plt.plot(res[1,:]*1e3,label="inhibitory")
    plt.plot(res[2,:],label="adaptation excitatory")
    plt.plot(res[3,:],label="adaptation inhibition")
    plt.plot(f_ext*1e3/(300/k),label='external input')
    plt.legend()

    sim=run_test(fe.size,P,fe,fi,f_ext,adapation,function)
    plt.figure()
    plt.plot(sim[0,:]*1e3,label="excitatory")
    plt.plot(sim[1,:]*1e3,label="inhibitory")
    plt.plot(sim[2,:],label="adaptation excitatory")
    plt.plot(sim[3,:],label="adaptation inhibition")
    plt.plot(f_ext*1e3/(300/k),label='external input')
    plt.legend()
    plt.show()
    return P


# name_file = "/home/kusch/Documents/project/co_simulation/co-simulation_mouse/test_nest/test_file/network_sim/bernoulli/network_b_10.0.npy"
# print("network")
# P = engin(parameters_ex=excitatory,parameters_in=inhibitory,k=1,name_file=name_file)
# print("'P_e':",np.array2string(P[:10], separator=', '),",",sep='')
# print("'P_i':",np.array2string(P[10:], separator=', '),",",sep='')
# name_file = "/home/kusch/Documents/project/co_simulation/co-simulation_mouse/test_nest/test_file/network_sim/noise_pop/network_b_9.0.npy"
# P = engin(parameters_ex=excitatory,parameters_in=inhibitory,k=int(excitatory['N_tot']*(1-excitatory['g'])*excitatory['p_connect']),name_file=name_file)
# print("'P_e':",np.array2string(P[:10], separator=', '),",",sep='')
# print("'P_i':",np.array2string(P[10:], separator=', '),",",sep='')

name_file = "/home/kusch/Documents/project/co_simulation/co-simulation_mouse/test_nest/network_b_6.0.npy"
P = engin(parameters_ex=excitatory,parameters_in=inhibitory,k=int(excitatory['N_tot']*(1-excitatory['g'])*excitatory['p_connect']),name_file=name_file,test1=test1)
print("'P_e':",np.array2string(P[:10], separator=', '),",",sep='')
print("'P_i':",np.array2string(P[10:], separator=', '),",",sep='')

