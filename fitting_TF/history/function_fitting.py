import nest
import os
import subprocess
import sys
import itertools
from scipy.optimize import minimize
import scipy.special as sp_spec
import numpy as np
# from nest_elephant_tvb.simulation.file_tvb.Zerlaut import ZerlautAdaptationSecondOrder as model
from nest_elephant_tvb.Tvb import Matteo_2 as model
from nest_elephant_tvb.Tvb import excitatory


# Matteo function
# # excitatory
# excitatory={
#         'C_m':200.0,
#         't_ref':5.0,
#         'V_reset':-64.5,
#         'E_L':-64.5,
#         'g_L':10.0,
#         'I_e':0.0,
#         'a':0.0,
#         'b':0.0,
#         'Delta_T':2.0,
#         'tau_w':500.0,
#         'V_th':-50.0,
#         'E_ex':0.0,
#         'tau_syn_ex':5.0,
#         'E_in':-80.0,
#         'tau_syn_in':5.0,
#     'V_peak': 10.0,
#     'N_tot':10**4,
#     'p_connect':0.05,
#     'g':0.2,
#     'Q_e':1.0,
#     'Q_i':2.5,
# }
# #inhibitory
# inhibitory={
#         'C_m':200.0,
#         't_ref':5.0,
#         'V_reset':-65.0,
#         'E_L':-65.,
#         'g_L':10.0,
#         'I_e':0.0,
#         'a':0.0,
#         'b':0.0,
#         'Delta_T':0.5,
#         'tau_w':1.0,
#         'V_th':-50.0,
#         'E_ex':0.0,
#         'tau_syn_ex':5.0,
#         'E_in':-80.0,
#         'tau_syn_in':5.0,
#     'V_peak': 10.0,
#     'N_tot':10**4,
#     'p_connect':0.05,
#     'g':0.2,
#     'Q_e':1.0,
#     'Q_i':2.5
# }


def compute_rate(data,begin,end,nb):
    """
    Compute the firing rate
    :param data: the spike of all neurons between end and begin
    :param begin: the time of the first spike
    :param end: the time of the last spike
    :return: the mean and the standard deviation of firing rate, the maximum and minimum of firing rate
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
    :param path: the path to the file
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

def create_transfer_function(parameter,excitatory):
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
    model_test.p_connect = np.array(parameter['p_connect'])
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

def effective_Vthre(Y, muV, sV, Tv):
    Vthre_eff = muV+np.sqrt(2)*sV*sp_spec.erfcinv(Y*2.*Tv) # effective threshold
    return Vthre_eff

def engin(parameters,excitatory,max_frequency=40.0,precision=0.5,frequency=None,rescale=None):
    name_file ='/home/kusch/Documents/project/co_simulation/co-simulation_mouse/test_nest/test_file/fitting/'
    for name,value in parameters.items():
        name_file += name+'_'+str(value)+'/'
    if frequency is None:
        frequency = np.arange(0.0,max_frequency,precision)
    frequencies = np.array(list(itertools.product(frequency,frequency)))

    if os.path.exists(name_file+'/P.npy'):
        return np.load(name_file+'/P.npy')
    elif os.path.exists(name_file+'/rate.npy'):
        rate = np.load(name_file+'/rate.npy')
    else:
        if os.path.exists(name_file+'/spike_detector.gdf'):
            print('analysis')
            data = load_spike(name_file)

        else:
            if not os.path.exists(name_file):
                os.makedirs(name_file)
            #initialisation of the parameter
            params = {      'g_L':parameters['g_L'],
                            'E_L':parameters['E_L'],
                            'V_reset':parameters['V_reset'],
                            'I_e':parameters['I_e'],
                            'C_m':parameters['C_m'],
                            'V_th':parameters['V_th'],
                            't_ref':parameters['t_ref'],
                            'tau_w':parameters['tau_w'],
                            'Delta_T':parameters['Delta_T'],
                            'b':parameters['b'],
                            'a':parameters['a'],
                            'V_peak':parameters['V_peak'],
                            'E_ex':parameters['E_ex'],
                            'E_in':parameters['E_in'],
                            'tau_syn_ex':parameters['tau_syn_ex'],
                            'tau_syn_in':parameters['tau_syn_in'],
                            'gsl_error_tol':1e-8
                            }
            Number_connexion_ex = parameters['N_tot']*parameters['p_connect']*(1-parameters['g'])
            Number_connexion_in = parameters['N_tot']*parameters['p_connect']*parameters['g']
            simtime=100000.0
            master_seed = 5
            local_num_threads = 8
            # simulation
            simulation = False
            error = 1.0e-6

            while error > 1.0e-20 and not simulation:
                params['gsl_error_tol'] = error
                # initialisation of nest
                nest.ResetKernel()
                nest.SetKernelStatus({
                    # Resolution of the simulation (in ms).
                    "resolution": 0.05,
                    # Print the time progress, this should only be used when the simulation
                    # is run on a local machine.
                    "print_time": True,
                    # If True, data will be overwritten,
                    # If False, a NESTError is raised if the files already exist.
                    "overwrite_files": True,
                    # Number of threads per MPI process.
                    'local_num_threads': local_num_threads,
                    # Path to save the output data
                    'data_path':  name_file,
                    # Masterseed for NEST and NumPy
                    'grng_seed': master_seed + local_num_threads,
                    # Seeds for the individual processes
                    'rng_seeds': range(master_seed + 1 + local_num_threads, master_seed + 1 + (2 * local_num_threads)),
                    })

                #create the network
                nest.SetDefaults('aeif_cond_exp', params)
                neurons = nest.Create('aeif_cond_exp', frequency.shape[0]**2)
                poisson_generator_ex = nest.Create('poisson_generator', frequency.shape[0])
                poisson_generator_in = nest.Create('poisson_generator', frequency.shape[0])
                nest.SetStatus(poisson_generator_ex,'rate',frequency*Number_connexion_ex)
                nest.SetStatus(poisson_generator_in,'rate',frequency*Number_connexion_in)
                nest.CopyModel("static_synapse", "excitatory",
                               {"weight": parameters['Q_e'], "delay": 1.0})
                nest.CopyModel("static_synapse", "inhibitory",
                               {"weight": -parameters['Q_i'], "delay": 1.0})
                for inh in range(len(frequency)):
                    for ex in range(len(frequency)):
                        nest.Connect(poisson_generator_ex[ex],neurons[ex+inh*len(frequency)],syn_spec="excitatory")
                        nest.Connect(poisson_generator_in[inh],neurons[ex+inh*len(frequency)],syn_spec="inhibitory")

                #create spike detector
                spikes_dec = nest.Create("spike_detector")
                nest.SetStatus(spikes_dec, [{"label": "spike",
                                          # "withtime": True,
                                          # "withgid": True,
                                          # "to_file": True,
                                            "record_to":"ascii",
                                           }])
                nest.Connect(neurons,spikes_dec)
                try :
                    nest.Simulate(simtime)
                    simulation = True
                    print('end')
                except nest.NESTError as exception:
                    template = "An exception of type {0} occurred. Arguments:\n{1!r}"
                    message = template.format(type(exception).__name__, exception.args)
                    print(message)
                    error = error/10.0
            print('analysis')

            # Concatenate the different spike files
            if subprocess.call([os.path.join(os.path.dirname(__file__),'script.sh'),name_file]) == 1:
                sys.stderr.write('ERROR bad concatenation of spikes file\n')
                exit(1)

            #Compute rate
            # data = load_event(nest.GetStatus(spikes_dec)[0]['events'] )
            data = load_spike(name_file)

        if data is None:
            print('compute rate')
            rate = np.zeros_like(frequencies)
            return [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]
        else:
            rate = compute_rate(data,0.0,100000.0,len(frequencies))
        np.save(name_file+'/rate.npy', rate)
        del data

    nb_freq = len (frequency)
    if rescale is not None:
        index = np.arange(0,nb_freq,1)
        index = index[np.where(np.logical_not(np.isin(index,rescale)))]
        rate = rate.reshape(nb_freq,nb_freq)
        rate = np.ravel(rate[index,:][:,index])
        frequencies =  np.array(list(itertools.product(frequency[index],frequency[index])))
        nb_freq -=len(rescale)

    # data = np.load('/home/kusch/Documents/project/Zerlaut/travail/Zerlaut/mean_field_for_multi_input_integration/transfer_functions/data/FS-cell_CONFIG1.npy')
    # data = np.load('/home/kusch/Documents/project/Zerlaut/travail/Zerlaut/mean_field_for_multi_input_integration/transfer_functions/data/RS-cell_CONFIG1.npy')
    # rate = np.ravel(data[0])*1e-3
    # frequencies = np.empty((len(rate),2))
    # frequencies[:,1] = np.ravel(data[2])
    # frequencies[:,0] = np.ravel(np.repeat([data[3]],data[2].shape[1],axis=0))

    #Compute mean of value for the model
    # rate+=1e-10
    muV, sV, Tv =model.get_fluct_regime_vars(frequencies[:,1]*1e-3,frequencies[:,0]*1e-3,0.00,0.0,0.0,parameters['Q_e'],parameters['tau_syn_ex'],parameters['E_ex'],parameters['Q_i'],parameters['tau_syn_in'],parameters['E_in'],
                          parameters['g_L'],parameters['C_m'],parameters['E_L'],parameters['N_tot'],parameters['p_connect'],parameters['g'],0.0,0.0)
    Tv+= parameters['g_L']/parameters['C_m']
    i_non_zeros = np.where((rate>=1/100000.0)  &(rate*Tv<1.0))
    # i_non_zeros=np.arange(0,len(rate),1)

    Vthre_eff = effective_Vthre(rate[i_non_zeros], muV[i_non_zeros], sV[i_non_zeros], Tv[i_non_zeros])*1e-3
    TvN = Tv[i_non_zeros]*parameters['g_L']/parameters['C_m']

    import matplotlib.pylab as plt
    # all = frequency[index]
    all =frequency
    index = i_non_zeros[0]; not_index = np.where(np.logical_not(np.isin(np.arange(0,nb_freq*nb_freq,1),i_non_zeros)))[0];
    fig = plt.figure();ax = fig.add_subplot(111, projection='3d');ax.scatter(muV, sV, Tv*parameters['g_L']/parameters['C_m'], marker='x',s=0.1);ax.set_xlabel('Vm');ax.set_ylabel('sV');ax.set_zlabel('Tv')
    fig = plt.figure();ax = fig.add_subplot(111, projection='3d');ax.scatter(muV[index], sV[index], Tv[index]*parameters['g_L']/parameters['C_m'], marker='x',s=0.1);ax.set_xlabel('Vm');ax.set_ylabel('sV');ax.set_zlabel('Tv');ax.scatter(muV[not_index], sV[not_index], Tv[not_index]*parameters['g_L']/parameters['C_m'], marker='o',s=0.1);
    plt.figure(); plt.plot(all,muV.reshape(nb_freq,nb_freq),'x',markersize=0.5);
    plt.figure(); plt.plot(all,sV.reshape(nb_freq,nb_freq),'x',markersize=0.5);
    plt.figure(); plt.plot(all,Tv.reshape(nb_freq,nb_freq)*parameters['g_L']/parameters['C_m'],'x',markersize=0.5)
    plt.figure(); plt.plot(index,TvN,'bx',markersize=0.5); plt.plot(not_index,Tv[not_index]*parameters['g_L']/parameters['C_m'],'rx',markersize=0.5)
    plt.figure(); plt.plot(index,muV[index],'bx',markersize=0.5); plt.plot(not_index,muV[not_index],'rx',markersize=0.5)
    plt.figure(); plt.plot(index,sV[index],'bx',markersize=0.5); plt.plot(not_index,sV[not_index],'rx',markersize=0.5)
    plt.figure();plt.plot(Vthre_eff,rate[index],'x',markersize=0.5);
    plt.figure();plt.plot(Vthre_eff,muV[index],'x',markersize=0.5);
    plt.figure();plt.plot(Vthre_eff,sV[index],'x',markersize=0.5);
    plt.figure();plt.plot(Vthre_eff,TvN,'x',markersize=0.5);
    plt.figure(); plt.plot(frequencies[:,1].reshape(nb_freq,nb_freq).transpose(),rate.reshape(nb_freq,nb_freq).transpose()*1e3)
    plt.show()


    TF = create_transfer_function(parameters,excitatory=excitatory)
    P = np.zeros(20)
    P[:5] = Vthre_eff.mean(), 1e-3, 1e-3, 1e-3, 1e-3

    def Res(p):
        pp=p
        vthre = model.threshold_func(muV[i_non_zeros], sV[i_non_zeros], TvN, *pp)
        return np.mean((Vthre_eff-vthre)**2)
        # return np.mean(np.abs(Vthre_eff-vthre)*1e3)
        # return np.mean(np.abs((Vthre_eff - vthre) / Vthre_eff))
    plsq = minimize(Res, P, method='SLSQP',options={'ftol': 1e-15, 'disp': True, 'maxiter':40000})
    # plsq = minimize(Res, P, method='SLSQP',tol=1e-10,\
    #                 options={'ftol': 1e-12, 'eps':1e-8,'disp': True, 'maxiter':100000})

    P = plsq.x
    def Res_1(p):
        return np.mean((rate[i_non_zeros] - TF(frequencies[i_non_zeros,1]*1e-3,frequencies[i_non_zeros,0]*1e-3,p)) ** 2)
        # return np.mean(np.abs((rate[i_non_zeros] - TF(frequencies[i_non_zeros,1]*1e-3,frequencies[i_non_zeros,0]*1e-3,p)))*1e3)
        # return np.mean(np.abs( (rate[i_non_zeros] - TF(frequencies[i_non_zeros, 1] * 1e-3, frequencies[i_non_zeros, 0] * 1e-3, p)) / rate[i_non_zeros]))
    # plsq = minimize(Res_1, P, method='nelder-mead',options={'xtol': 1e-5, 'disp': True, 'maxiter': 50000})
    # plsq = minimize(Res_1, P, method='nelder-mead',tol=1e-12, \
    #                 options={'xtol': 1., 'disp': True, 'maxiter': 100000, 'maxfev':100000})
    # p = plsq.x
    # index = np.argsort(np.abs(((rate - TF(frequencies[:, 1] * 1e-3, frequencies[:, 0] * 1e-3, p)) * 1e3)))[-10:]
    # print('first part ')
    # print(P)
    # print("frequency", frequencies[index])
    # print("expected : ", rate[index] * 1e3)
    # print("got : ", TF(frequencies[:, 1] * 1e-3, frequencies[:, 0] * 1e-3, p)[index] * 1e3)
    # print("error : ", np.abs(((rate - TF(frequencies[:, 1] * 1e-3, frequencies[:, 0] * 1e-3, p)) * 1e3))[index])
    # print("max error ", np.abs(((rate - TF(frequencies[:, 1] * 1e-3, frequencies[:, 0] * 1e-3, p)) * 1e3))[index[-1]])

    P = plsq.x
    def Res_2(p):
        return np.mean((rate - TF(frequencies[:,1]*1e-3,frequencies[:,0]*1e-3,p)) ** 2)
        # return np.mean(np.abs((rate - TF(frequencies[:,1]*1e-3,frequencies[:,0]*1e-3,p)))*1e3)
        # return np.mean(np.abs((rate - TF(frequencies[:,1]*1e-3,frequencies[:,0]*1e-3,p))/rate))
    plsq = minimize(Res_2, P, method='nelder-mead',options={'xtol': 1e-5, 'disp': True, 'maxiter': 50000})
    # plsq = minimize(Res_2, P, method='nelder-mead',tol=1e-12, \
    #                 options={'xtol': 1.0, 'disp': True, 'maxiter': 100000, 'maxfev':100000})

    # network = np.empty((651,9))
    # range_rate = np.arange(0.0,105.0,5.0)
    # range_b =  np.arange(0.,3.1,0.1)
    # range_test = np.array(list(itertools.product(range_b,range_rate)))
    # for i in range(651):
    #     network[i,:7]=np.load("/home/kusch/Documents/project/co_simulation/co-simulation_mouse/test_nest/test_file/fitting/network_"+str(i)+".npy")
    # network[:,7:]=range_test
    # network = network[np.where(network[:,6]==0)[0],:]
    # index_network = 2 if excitatory else 0
    # P = plsq.x
    # def Res_3(p):
        # return np.mean((( network[:,index_network]*1e-3- TF(network[:,2]*1e-3,network[:,0]*1e-3,p,network[:,8]*1e-3,0.0,network[:,4]))*1e3) ** 2)
        # return np.mean(np.abs( network[:,index_network]- TF(network[:,2]*1e-3,network[:,0]*1e-3,p,network[:,8]*1e-3,0.0,network[:,4])*1e3))
        # return np.mean(np.abs( (network[:,index_network] - TF(network[:,2]*1e-3,network[:,0]*1e-3,p,network[:,8]*1e-3,0.0,network[:,4])*1e3))/network[:,index_network])
    # plsq = minimize(Res_3, P, method='nelder-mead',tol=1e-12, \
    #             options={'xtol': 1.0, 'disp': True, 'maxiter': 100000, 'maxfev':100000})

    p = plsq.x
    index = np.argsort(np.abs(((rate - TF(frequencies[:, 1] * 1e-3, frequencies[:, 0] * 1e-3, p)) * 1e3)))[-10:]
    # print("frequency", frequencies[index])
    # print("expected : ", rate[index] * 1e3)
    # print("got : ", TF(frequencies[:, 1] * 1e-3, frequencies[:, 0] * 1e-3, p)[index] * 1e3)
    # print("error : ", np.abs(((rate - TF(frequencies[:, 1] * 1e-3, frequencies[:, 0] * 1e-3, p)) * 1e3))[index])
    # print("max error ", np.abs(((rate - TF(frequencies[:, 1] * 1e-3, frequencies[:, 0] * 1e-3, p)) * 1e3))[index[-1]])
    # np.save(name_file+'/P.npy', plsq.x)



    TF(frequencies[:,1]*1e-3,frequencies[:,0]*1e-3,p)
    plt.figure(); plt.plot(frequencies[:,1].reshape(nb_freq,nb_freq).transpose(),TF(frequencies[:,1]*1e-3,frequencies[:,0]*1e-3,p).reshape(nb_freq,nb_freq).transpose()*1e3)
    plt.figure(); plt.plot(frequencies[:,1].reshape(nb_freq,nb_freq).transpose(),rate.reshape(nb_freq,nb_freq).transpose()*1e3)
    plt.show()
    return plsq.x



# frequency = np.array([0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.61,0.62,0.63,0.64,0.65,0.66,0.67,0.68,0.69,0.7,0.71,0.72,0.73,0.74,0.75,0.76,0.77,0.78,0.79,0.8,0.81,0.82,0.83,0.84,0.85,0.86,0.87,0.88,0.89,0.9,0.92,0.95,0.97,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,2.2,2.5,2.7,3.0,3.2,3.5,3.7,4.0,4.2,4.5,4.7,5.0,5.2,5.7,6.0,6.5,7.0,7.5,8.0,8.5,9.0,
#              10.0,11.0,12.0,13.0,14.0,15.0,16.0,17.0,18.0,19.0,20.0,21.0,22.0,23.0,24.0,25.0,26.0,27.0,28.0,29.0,30.0,31.0,32.0,33.0,34.0,35.0,36.0,37.0,38.0,39.0,40.0,
#                       50.0,60.0])
#                       41.0,42.0,43.0,44.0,45.0,46.0,47.0,48.0,49.0,50.0,51.0,52.0,53.0,54.0,55.0,56.0,57.0,58.0,59.0,60.0])
                      # 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0,
                      # 57.0, 58.0, 59.0, 60.0,61.0,62.0,63.0,64.0,65.0,66.0,67.0,68.0,69.0,70.0,71.0,72.0,73.0,74.0,75.0,76.0,77.0,78.0,79.0,80.0,
                      # 85.0,90.0,100.0,110.0,120.0,130.0,140.0,150.0,160.0,170.0,180.0,200.0])

print("EXCITATORY")
print("'P_e':",np.array2string(engin(parameters=excitatory,excitatory=True,max_frequency=40.0,precision=0.5), separator=', '),",",sep='')
# print("'P_e':",np.array2string(engin(parameters=excitatory,excitatory=True,max_frequency=40.0,precision=0.5,rescale=[32,33,56,57,59,60,62,63]), separator=', '),",",sep='')
print("INHIBITORY")
# print("'P_i':",np.array2string(engin(parameters=inhibitory,excitatory=False,max_frequency=40.0,precision=0.5), separator=', '),",",sep='')
# print("'P_i':",np.array2string(engin(parameters=inhibitory,excitatory=False,max_frequency=40.0,precision=0.5,rescale=[5,6]), separator=', '),",",sep='')
# print(engin(parameters=default,frequency=frequency))
# print(engin(parameters=default,max_frequency=60.0,precision=0.5))
# print(engin(parameters=default,max_frequency=100.0,precision=0.5))
# np.set_printoptions(linewidth=500)
# int_first = 200
print("EXCITATORY")
# print("'P_e':",np.array2string(engin(parameters=excitatory,excitatory=True,max_frequency=100.0,precision=0.5,rescale=[10,12,123,124,174]), separator=', '),",",sep='')
# rescale = np.arange(int_first,200)
# print("'P_e':",np.array2string(engin(parameters=excitatory,excitatory=True,max_frequency=100.0,precision=0.5,rescale=rescale), separator=', '),",",sep='')
# print("'P_e':",np.array2string(engin(parameters=excitatory,max_frequency=40.0,precision=0.5), separator=', '),",",sep='')
print("INHIBITORY")
# print("'P_i':",np.array2string(engin(parameters=inhibitory,excitatory=False,max_frequency=100.0,precision=0.5,rescale=[3,4,44,45,59,60,81,82,105,106,109,110,111,112,113,114,194,195]), separator=', '),",",sep='')
# print("'P_i':",np.array2string(engin(parameters=inhibitory,excitatory=False,max_frequency=100.0,precision=0.5,rescale=[44,45,59,60,81,82,105,106,109,110,111,112,194,195]), separator=', '),",",sep='')
# rescale = np.arange(int_first,200)
# print("'P_i':",np.array2string(engin(parameters=inhibitory,excitatory=False,max_frequency=100.0,precision=0.5,rescale=rescale), separator=', '),",",sep='')
# print("'P_i':",np.array2string(engin(parameters=inhibitory,max_frequency=40.0,precision=0.5), separator=', '),",",sep='')


def model_end (g_L,E_L,V_reset,I_e,C_m,V_th,t_ref,tau_w,Delta_T,b,a,E_ex,E_in,tau_syn_ex,tau_syn_in,Q_e,Q_i,N_tot,p_connect,g):
    parameters= {   'g_L':g_L,
                    'E_L':E_L,
                    'V_reset':V_reset,
                    'I_e':I_e,
                    'C_m':C_m,
                    'V_th':V_th,
                    't_ref':t_ref,
                    'tau_w':tau_w,
                    'Delta_T':Delta_T,
                    'b':b,
                    'a':a,
                    'V_peak':-10.0,
                    'E_ex':E_ex,
                    'E_in':E_in,
                    'tau_syn_ex':tau_syn_ex,
                    'tau_syn_in':tau_syn_in,
                    'Q_e': Q_e,
                    'Q_i' : Q_i,
                    'N_tot' : N_tot,
                    'p_connect' : p_connect,
                    'g': g
                    }

    return None, engin(parameters=parameters,max_frequency=40.0)