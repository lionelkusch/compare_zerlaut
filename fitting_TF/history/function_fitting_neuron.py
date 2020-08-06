import nest
import os
from scipy.optimize import minimize
import scipy.special as sp_spec
import numpy as np
test1=False
from nest_elephant_tvb.Tvb.modify_tvb.Zerlaut import ZerlautAdaptationSecondOrder as model
# test1=True
# from nest_elephant_tvb.simulation.file_tvb.Zerlaut_test_1 import ZerlautAdaptationSecondOrder as model
import logging
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)

nest.set_verbosity(100)

# excitatory
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
    # 'Q_i':5.0,
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
    # 'Q_i':5.0
}


def compute_rate(data,begin,end,nb):
    """
    Compute the firing rate
    :param data: the spike of all neurons between end and begin
    :param begin: the time of the first spike
    :param end: the time of the last spike
    :return: the mean and the standard deviation of firing rate, the maximum and minimum of firing rate
    """
    #get data
    n_fil = data[:]
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

def engin(parameters,excitatory,
          MAXfexc=40., MINfexc=0., nb_value_fexc=60,
          MAXfinh=40., MINfinh=0., nb_value_finh=20,
          MAXadaptation=200.,MINadaptation=0., nb_value_adaptation=20,
          MAXfout=20., MINfout=0.1, MAXJump=1.0, MINJump=0.1,
          SEED=50,
          name_file='/home/kusch/Documents/project/co_simulation/co-simulation_mouse/test_nest/test_file/fitting/',
          dt=1e-4, tstop=10.0,
          test1=False):
    # for name,value in parameters.items():
    #     name_file += name+'_'+str(value)+'/'
    if excitatory:
        name_file = '/home/kusch/Documents/project/co_simulation/co-simulation_mouse/test_nest/test_file/fitting/a_0.0/E_in_-80.0/Q_e_1.0/b_0.0/V_th_-50.0/g_0.2/Delta_T_2.0/I_e_0.0/Q_i_3.5/C_m_200.0/tau_syn_ex_5.0/N_tot_10000/g_L_10.0/p_connect_0.05/E_ex_0.0/V_reset_-64.5/tau_syn_in_5.0/tau_w_500.0/E_L_-64.5/V_peak_10.0/t_ref_5.0/'
    else:
        name_file = '/home/kusch/Documents/project/co_simulation/co-simulation_mouse/test_nest/test_file/fitting/a_0.0/E_in_-80.0/Q_e_1.0/b_0.0/V_th_-50.0/g_0.2/Delta_T_0.5/I_e_0.0/Q_i_3.5/C_m_200.0/tau_syn_ex_5.0/N_tot_10000/g_L_10.0/p_connect_0.05/E_ex_0.0/V_reset_-65.0/tau_syn_in_5.0/tau_w_1.0/E_L_-65.0/V_peak_10.0/t_ref_5.0/'

    if os.path.exists(name_file+'/P.npy'):
        return np.load(name_file+'/P.npy')
    elif os.path.exists(name_file+'/fout.npy'):
        feOut = np.mean(np.load(name_file+'/fout.npy'),axis=2).ravel()
        feSim = np.load(name_file+'/fin.npy').ravel()
        fiSim = np.repeat([np.repeat(np.linspace(MINfinh,MAXfinh, nb_value_finh),nb_value_adaptation)],nb_value_fexc,axis=0).ravel()
        adaptation = np.repeat([np.repeat([np.linspace(MINadaptation, MAXadaptation, nb_value_adaptation)],nb_value_finh,axis=0)],nb_value_fexc,axis=0).ravel()
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
        simtime=tstop*1e3
        dt=dt*1e3
        master_seed = 0
        local_num_threads = 8

        # intialisation of variable
        fiSim = np.repeat(np.linspace(MINfinh,MAXfinh, nb_value_finh),nb_value_adaptation).reshape(nb_value_finh*nb_value_adaptation)*Number_connexion_in
        adaptation = np.repeat([np.linspace(MINadaptation, MAXadaptation, nb_value_adaptation)],nb_value_finh,axis=0).reshape(nb_value_finh*nb_value_adaptation)
        feSim =  np.zeros((nb_value_fexc,nb_value_finh*nb_value_adaptation))
        feOut = np.zeros((nb_value_fexc,nb_value_finh*nb_value_adaptation,SEED))
        MAXdfex =  (MAXfexc-MINfexc)/nb_value_fexc
        dFex = np.ones((nb_value_adaptation*nb_value_finh))*MAXdfex
        index_end = np.zeros((nb_value_adaptation*nb_value_finh),dtype=np.int)
        index = np.where(index_end >= 0)
        while index[0].size > 0:
            step = np.min(index_end[index])
            index_min = index[0][np.argmin(index_end[index])]
            print(step,index_min,dFex[index_min],feOut[step-1,index_min,:],feOut[step,index_min,:],np.mean(feOut[step-1,index_min]),np.mean(feOut[step,index_min]),feSim[step,index_min],fiSim[index_min],adaptation[index_min])

            # simulation
            simulation = False
            error = 1.0e-6

            while error > 1.0e-20 and not simulation:
                params['gsl_error_tol'] = error
                # initialisation of nest
                nest.ResetKernel()
                nest.SetKernelStatus({
                    # Resolution of the simulation (in ms).
                    "resolution": dt,
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
                neurons = nest.Create('aeif_cond_exp', index[0].size*SEED)
                nest.SetStatus(neurons,"I_e", -np.repeat(adaptation[index],SEED).ravel())
                poisson_generator_ex = nest.Create('poisson_generator', index[0].size*SEED)
                poisson_generator_in = nest.Create('poisson_generator', index[0].size*SEED)
                nest.SetStatus(poisson_generator_in,'rate',np.repeat(fiSim[index],SEED).ravel())
                nest.SetStatus(poisson_generator_ex,'rate',np.repeat(feSim[index_end[index].ravel(),index],SEED)*Number_connexion_ex)
                nest.CopyModel("static_synapse", "excitatory",
                               {"weight": parameters['Q_e'], "delay": 1.0})
                nest.CopyModel("static_synapse", "inhibitory",
                               {"weight": -parameters['Q_i'], "delay": 1.0})
                nest.Connect(poisson_generator_ex,neurons,'one_to_one',syn_spec="excitatory")
                nest.Connect(poisson_generator_in,neurons,'one_to_one',syn_spec="inhibitory")

                #create spike detector
                spikes_dec = nest.Create("spike_detector")
                nest.Connect(neurons,spikes_dec)
                try :
                    nest.Simulate(simtime)
                    simulation = True
                except nest.NESTError as exception:
                    template = "An exception of type {0} occurred. Arguments:\n{1!r}"
                    message = template.format(type(exception).__name__, exception.args)
                    print(message)
                    error = error/10.0
            # compute firing rate
            data = nest.GetStatus(spikes_dec)[0]['events']['senders']
            feOut[index_end[index].ravel(),index] = compute_rate(data,0.0,simtime,index[0].size*SEED).reshape(index[0].size,SEED)*1e3
            jump = np.mean(feOut[index_end.ravel(),np.arange(0,index_end.size)],axis=1) - np.mean(feOut[(index_end-1).ravel(),np.arange(0,index_end.size)],axis=1)

            # rescale if jump to big
            update_index = np.where(np.logical_and(index_end>=0, jump > MAXJump))
            feSim[index_end.ravel()[update_index],update_index] -=dFex[update_index]
            dFex[update_index]/=2
            feSim[index_end.ravel()[update_index],update_index] +=dFex[update_index]

            # increase external input if no spike (initial condition  of external input)
            update_index = np.where(np.logical_and(np.logical_and(index_end>=0, jump <= MAXJump),jump < MINJump))
            feSim[index_end.ravel()[update_index],update_index]+=dFex[update_index]
            dFex[update_index]+=dFex[update_index]*0.1

            # save the data and pass at next value
            update_index = np.where(np.logical_and(np.logical_and(index_end>=0, jump <= MAXJump),jump >= MINJump))
            index_end[update_index]+=1
            index_end[np.where(index_end==nb_value_fexc)[0]]=-1

            update =np.where(np.logical_and(np.logical_and(index_end>=0, jump <= MAXJump),jump > MINJump))
            feSim[index_end.ravel()[update],update]=feSim[index_end.ravel()[update]-1,update]+dFex[update]
            update =np.where(np.logical_and(np.logical_and(index_end>=0, jump <= MAXJump), dFex<MAXdfex))[0]
            dFex[update][np.where(dFex[update] < MAXdfex)]+=dFex[update]*0.1
            index = np.where(index_end >= 0)

        np.save(name_file+'/fout.npy',feOut)
        np.save(name_file+'/fin.npy',feSim)

    mask = np.where(feOut < MAXfexc)
    feOut = feOut[mask]*1e-3
    feSim = feSim[mask]*1e-3
    fiSim = fiSim[mask]*1e-3
    adaptation = adaptation[mask]
    #Compute mean of value for the model
    muV, sV, Tv =model.get_fluct_regime_vars(feSim,fiSim,0.00,0.0,adaptation,parameters['Q_e'],parameters['tau_syn_ex'],parameters['E_ex'],parameters['Q_i'],parameters['tau_syn_in'],parameters['E_in'],
                          parameters['g_L'],parameters['C_m'],parameters['E_L'],parameters['N_tot'],parameters['p_connect'],parameters['g'],0.0,0.0)
    Tv+= parameters['g_L']/parameters['C_m']
    i_non_zeros = np.where(feOut*Tv<1.0)

    Vthre_eff = effective_Vthre(feOut[i_non_zeros], muV[i_non_zeros], sV[i_non_zeros], Tv[i_non_zeros])*1e-3
    TvN = Tv[i_non_zeros]*parameters['g_L']/parameters['C_m']

    TF = create_transfer_function(parameters,excitatory=excitatory)
    if test1:
        P = np.zeros(15)
    else:
        P = np.zeros(10)
    P[:5] = Vthre_eff.mean(), 1e-3, 1e-3, 1e-3, 1e-3

    def Res(p):
        pp=p
        if test1:
            vthre = model.threshold_func(muV[i_non_zeros], sV[i_non_zeros], TvN, adaptation[i_non_zeros], *pp)
        else:
            vthre = model.threshold_func(muV[i_non_zeros], sV[i_non_zeros], TvN,  *pp)
        return np.mean((Vthre_eff-vthre)**2)
    plsq = minimize(Res, P, method='SLSQP',options={
                                                    'ftol': 1e-15,
                                                    'disp': True,
                                                    'maxiter':50000})

    P = plsq.x
    print("error ", np.mean(((feOut - TF(feSim, fiSim, P,w=adaptation)) ** 2) * 1e3))
    def Res_2(p):
        return np.mean(((feOut - TF(feSim,fiSim,p,w=adaptation))*1e3) ** 2)
    plsq = minimize(Res_2, P, method='nelder-mead',
                                                        tol=1e-11,
                                                        # tol=1e-7,
                                                        options={
                                                        'xatol': 1e-15,
                                                        'disp': True,
                                                        'maxiter': 50000})

    p_with = plsq.x
    # np.save(name_file+'/P.npy', plsq.x)

    # without adaptaition
    mask = np.where(adaptation == 0.0)
    feOut_1 = feOut[mask]
    feSim_1 = feSim[mask]
    fiSim_1 = fiSim[mask]
    adaptation_1 = adaptation[mask]
    muV, sV, Tv =model.get_fluct_regime_vars(feSim_1,fiSim_1,0.00,0.0,adaptation_1,parameters['Q_e'],parameters['tau_syn_ex'],parameters['E_ex'],parameters['Q_i'],parameters['tau_syn_in'],parameters['E_in'],
                          parameters['g_L'],parameters['C_m'],parameters['E_L'],parameters['N_tot'],parameters['p_connect'],parameters['g'],0.0,0.0)
    Tv+= parameters['g_L']/parameters['C_m']
    i_non_zeros = np.where(feOut_1*Tv<1.0)

    Vthre_eff = effective_Vthre(feOut_1[i_non_zeros], muV[i_non_zeros], sV[i_non_zeros], Tv[i_non_zeros])*1e-3
    TvN = Tv[i_non_zeros]*parameters['g_L']/parameters['C_m']

    TF = create_transfer_function(parameters,excitatory=excitatory)
    if test1:
        P = np.zeros(15)
    else:
        P = np.zeros(10)
    P[:5] = Vthre_eff.mean(), 1e-3, 1e-3, 1e-3, 1e-3

    def Res(p):
        pp=p
        if test1:
            vthre = model.threshold_func(muV[i_non_zeros], sV[i_non_zeros], TvN, adaptation_1,*pp)
        else:
            vthre = model.threshold_func(muV[i_non_zeros], sV[i_non_zeros], TvN, *pp)
        return np.mean((Vthre_eff-vthre)**2)
    plsq = minimize(Res, P, method='SLSQP',options={
                                                    'ftol': 1e-15,
                                                    'disp': True,
                                                    'maxiter':40000})

    P = plsq.x
    print("error ", np.mean(((feOut - TF(feSim, fiSim, P,w=adaptation)) ** 2) * 1e3))
    def Res_2(p):
        return np.mean((feOut_1 - TF(feSim_1,fiSim_1,p,w=adaptation_1)) ** 2)
    plsq = minimize(Res_2, P, method='nelder-mead',
                                                    tol=1e-11,
                                                    # tol=1e-7,
                                                    options={
                                                        'xtol': 1e-15,
                                                        'disp': True,
                                                        'maxiter': 50000})

    p_without = plsq.x

    i=0
    result_n = np.empty((nb_value_fexc,nb_value_finh,nb_value_adaptation,4))
    result_n[:]=np.NAN
    fe_model = -1
    np.linspace(MINadaptation, MAXadaptation, nb_value_adaptation)
    while i != len(fiSim):
        fi_model = np.where(fiSim[i]==np.linspace(MINfinh,MAXfinh, nb_value_finh)*1e-3)[0][0]
        w_model = np.where(adaptation[i]==np.linspace(MINadaptation, MAXadaptation, nb_value_adaptation))[0][0]
        if adaptation[i] < adaptation[i-1]:
            if fiSim[i] < fiSim[i-1]:
                fe_model += 1
        result_n[fe_model, fi_model, w_model, :] = [feOut[i], feSim[i], fiSim[i], adaptation[i]]

        i+=1
    import matplotlib.pyplot as plt
    # for i in range(nb_value_adaptation):
    #     fig, axs = plt.subplots(2, 2, figsize=(20, 20))
    #     axis= axs[0,0]
    #     mat = axis.imshow(result_n[:,:,i,0])
    #     axis.set_title('F_out')
    #     fig.colorbar(mat, ax=axis, extend='both')
    #     axis= axs[0,1]
    #     mat =axis.imshow(result_n[:,:,i,1])
    #     axis.set_title('Fe_in')
    #     fig.colorbar(mat, ax=axis, extend='both')
    #     axis= axs[1,0]
    #     mat =axis.imshow(result_n[:,:,i,2])
    #     axis.set_title('Fi_in')
    #     fig.colorbar(mat, ax=axis, extend='both')
    #     axis= axs[1,1]
    #     mat =axis.imshow(result_n[:,:,i,3])
    #     fig.colorbar(mat, ax=axis, extend='both')
    #     axis.set_title('adaptation')
    #     fig.suptitle('adaptation = '+str(adaptation[i]), fontsize=16)
    for i in range(nb_value_adaptation):
        fig=plt.figure(figsize=(20,20))
        plt.plot(result_n[:,:,i,1]*1e3,result_n[:,:,i,0]*1e3, '.g')
        plt.plot(result_n[:,:,i,1]*1e3,TF(result_n[:,:,i,1],result_n[:,:,i,2] , p_with,w=result_n[:,:,i,3])*1e3,'--c')
        plt.plot(result_n[:,:,i,1]*1e3,TF(result_n[:,:,i,1],result_n[:,:,i,2] , p_without,w=result_n[:,:,i,3])*1e3,'b')
        plt.title('adapt = '+str(result_n[:,:,i,3][0][0]),{"fontsize":30.0})
        plt.ylabel('output frequency of the neurons in Hz',{"fontsize":30.0})
        plt.xlabel('excitatory input frequency in Hz',{"fontsize":30.0})
        plt.tick_params(labelsize=10.0)
        np.set_printoptions(precision=2)
        plt.text(10, -7, "frequency inhibitory "+str(repr(result_n[:,:,i,2][0]*1e3)), ha='center', fontsize=20.0)
        for j in range(nb_value_finh):
            for k in range(nb_value_fexc):
                if not np.isnan(result_n[k,j,i,0]):
                    plt.plot([result_n[k,j,i,1]*1e3,result_n[k,j,i,1]*1e3],[result_n[k,j,i,0]*1e3,TF(result_n[k,j,i,1],result_n[k,j,i,2] , p_without,w=result_n[k,j,i,3])*1e3],color='r',alpha=0.5)
                    # plt.plot([result_n[k, j, i, 1], result_n[k, j, i, 1]], [result_n[k, j, i, 0], TF(result_n[k, j, i, 1], result_n[k, j, i, 2], p_with, w=result_n[k, j, i, 3])], color='r', alpha=0.5)
        if excitatory:
            name_fig = '/home/kusch/Documents/project/co_simulation/co-simulation_mouse/test_nest/test_file/fitting/ex' + str(i) + '.svg'
        else:
            name_fig = '/home/kusch/Documents/project/co_simulation/co-simulation_mouse/test_nest/test_file/fitting/inh' + str(i) + '.svg'
        plt.savefig(name_fig)
    # plt.show()
    np.set_printoptions(edgeitems=3,infstr='inf', linewidth=75, nanstr='nan', precision=8, suppress=False, threshold=1000, formatter=None)

    print("######################## with adaptation ######################")
    index = np.argsort(np.abs(((feOut - TF(feSim, fiSim, p_with,w=adaptation)) * 1e3)))[-5:]
    print("frequency ex", feSim[index]*1e3)
    print("frequency in", fiSim[index]*1e3)
    print("adaptation", adaptation[index])
    print("expected : ", feOut[index]*1e3)
    print("got : ", TF(feSim, fiSim, p_with,w=adaptation)[index] * 1e3)
    print("error : ", np.abs(((feOut - TF(feSim,fiSim, p_with,w=adaptation)) * 1e3))[index])
    print("max error ", np.abs(((feOut - TF(feSim, fiSim, p_with,w=adaptation)) * 1e3))[index[0]])
    print("error ", np.mean(((feOut - TF(feSim, fiSim, p_with,w=adaptation)) ** 2) * 1e3))
    print("                 #### no adaptation #####                 ")
    index = np.argsort(np.abs(((feOut_1 - TF(feSim_1, fiSim_1, p_with,w=adaptation_1)) * 1e3)))[-5:]
    print("frequency ex", feSim_1[index]*1e3)
    print("frequency in", fiSim_1[index]*1e3)
    print("adaptation", adaptation_1[index])
    print("expected : ", feOut_1[index]*1e3)
    print("got : ", TF(feSim_1, fiSim_1, p_with,w=adaptation_1)[index] * 1e3)
    print("error : ", np.abs(((feOut_1 - TF(feSim_1,fiSim_1, p_with,w=adaptation_1)) * 1e3))[index])
    print("max error ", np.abs(((feOut_1 - TF(feSim_1, fiSim_1, p_with,w=adaptation_1)) * 1e3))[index[0]])
    print("error ", np.mean(((feOut_1 - TF(feSim_1, fiSim_1, p_with,w=adaptation_1)) ** 2) * 1e3))
    print("######################## without adaptation ######################")
    index = np.argsort(np.abs(((feOut - TF(feSim, fiSim, p_without,w=adaptation)) * 1e3)))[-5:]
    print("frequency ex", feSim[index]*1e3)
    print("frequency in", fiSim[index]*1e3)
    print("adaptation", adaptation[index])
    print("expected : ", feOut[index]*1e3)
    print("got : ", TF(feSim, fiSim, p_without,w=adaptation)[index] * 1e3)
    print("error : ", np.abs(((feOut - TF(feSim,fiSim, p_without,w=adaptation)) * 1e3))[index])
    print("max error ", np.abs(((feOut - TF(feSim, fiSim, p_without,w=adaptation)) * 1e3))[index[0]])
    print("error ", np.mean(((feOut - TF(feSim, fiSim, p_without,w=adaptation)) ** 2) * 1e3))
    print("                 #### no adaptation #####                 ")
    index = np.argsort(np.abs(((feOut_1 - TF(feSim_1, fiSim_1, p_without,w=adaptation_1)) * 1e3)))[-5:]
    print("frequency ex", feSim_1[index]*1e3)
    print("frequency in", fiSim_1[index]*1e3)
    print("adaptation", adaptation_1[index])
    print("expected : ", feOut_1[index]*1e3)
    print("got : ", TF(feSim_1, fiSim_1, p_without,w=adaptation_1)[index] * 1e3)
    print("error : ", np.abs(((feOut_1 - TF(feSim_1,fiSim_1, p_without,w=adaptation_1)) * 1e3))[index])
    print("max error ", np.abs(((feOut_1 - TF(feSim_1, fiSim_1, p_without,w=adaptation_1)) * 1e3))[index[0]])
    print("error ", np.mean(((feOut_1 - TF(feSim_1, fiSim_1, p_without,w=adaptation_1)) ** 2) * 1e3))
    print("##############################################")
    print(p_without)
    print(p_with)
    return p_with


# print("'P_e':",np.array2string(engin(parameters=excitatory,excitatory=True), separator=', '),",",)
# print("'P_i':",np.array2string(engin(parameters=inhibitory,excitatory=False), separator=', '),",")
print("'P_e':",np.array2string(engin(parameters=excitatory,excitatory=True,test1=test1), separator=', '),",",sep='')
print("'P_i':",np.array2string(engin(parameters=inhibitory,excitatory=False,test1=test1), separator=', '),",",sep='')


