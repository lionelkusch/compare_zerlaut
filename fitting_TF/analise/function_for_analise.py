import nest_elephant_tvb.Tvb.modify_tvb.Zerlaut as Zerlaut
import numpy as np
import sys

def get_result(path,time_begin,time_end):
    '''
    return the result of the simulation between the wanted time
    :param path: the folder of the simulation
    :param time_begin: the start time for the result
    :param time_end:  the ending time for the result
    :return: result of all monitor
    '''
    sys.path.append(path)
    from parameter import param_tvb
    sys.path.remove(path)
    count_begin = int(time_begin/param_tvb['save_time'])
    count_end = int(time_end/param_tvb['save_time'])+1
    nb_monitor = param_tvb['Raw'] + param_tvb['TemporalAverage'] + param_tvb['Bold']
    output =[]

    for count in range(count_begin,count_end):
        result = np.load(path+'/step_'+str(count)+'.npy',allow_pickle=True)
        for i in range(result.shape[0]):
            tmp = np.array(result[i])
            if len(tmp) != 0:
                tmp = tmp[np.where((time_begin <= tmp[:,0]) &  (tmp[:,0]<= time_end)),:]
                tmp_time = tmp[0][:,0]
                if tmp_time.shape[0] != 0:
                    one = tmp[0][:,1][0]
                    tmp_value = np.concatenate(tmp[0][:,1]).reshape(tmp_time.shape[0],one.shape[0],one.shape[1])
                    if len(output) == nb_monitor:
                        output[i]=[np.concatenate([output[i][0],tmp_time]),np.concatenate([output[i][1],tmp_value])]
                    else:
                        output.append([tmp_time,tmp_value])
    return output

def print_bistability(param_tvb,param_zerlaut,param_nest,param_topology,param_connection,param_background):
    from scipy.optimize import fsolve
    '''
    print if the model is bistable or not
    :param parameter_model: parameters for the model
        (the parameter external_input_in_in and external_input_in_ex is taking in count)
    :return: nothing
    '''
    ## Model
    if param_zerlaut['order'] == 1:
        model = Zerlaut.ZerlautAdaptationFirstOrder(variables_of_interest='E I W_e W_i'.split())
    elif param_zerlaut['order'] == 2:
        model = Zerlaut.ZerlautAdaptationSecondOrder(variables_of_interest='E I C_ee C_ei C_ii W_e W_i'.split())
    else:
        raise Exception('Bad order for the model')

    model.g_L = np.array(param_topology['param_neuron_excitatory']['g_L'])
    model.E_L_e = np.array(param_topology['param_neuron_excitatory']['E_L'])
    model.E_L_i = np.array(param_topology['param_neuron_inhibitory']['E_L'])
    model.C_m = np.array(param_topology['param_neuron_excitatory']['C_m'])
    model.b_e = np.array(param_topology['param_neuron_excitatory']['b'])
    model.a_e = np.array(param_topology['param_neuron_excitatory']['a'])
    model.b_i = np.array(param_topology['param_neuron_inhibitory']['b'])
    model.a_i = np.array(param_topology['param_neuron_inhibitory']['a'])
    model.tau_w_e = np.array(param_topology['param_neuron_excitatory']['tau_w'])
    model.tau_w_i = np.array(param_topology['param_neuron_inhibitory']['tau_w'])
    model.E_e = np.array(param_topology['param_neuron_excitatory']['E_ex'])
    model.E_i = np.array(param_topology['param_neuron_excitatory']['E_in'])
    model.Q_e = np.array(param_connection['weight_local'])
    model.Q_i = np.array(param_connection['weight_local'] * param_connection['g'])
    model.tau_e = np.array(param_topology['param_neuron_excitatory']['tau_syn_ex'])
    model.tau_i = np.array(param_topology['param_neuron_excitatory']['tau_syn_in'])
    model.N_tot = np.array(param_topology['nb_neuron_by_region'])
    model.p_connect = np.array(param_connection['p_connect'])
    model.g = np.array(param_topology['percentage_inhibitory'])
    model.T = np.array(param_zerlaut['T'])
    model.P_e = np.array(param_zerlaut['P_e'])
    model.P_i = np.array(param_zerlaut['P_i'])
    model.K_ext_e = np.array(param_connection['nb_external_synapse'])
    model.K_ext_i = np.array(0)
    model.external_input_ex_ex = np.array(0.)
    model.external_input_ex_in = np.array(0.)
    model.external_input_in_ex = np.array(0.0)
    model.external_input_in_in = np.array(0.0)
    model.state_variable_range['E'] = np.array(param_zerlaut['initial_condition']['E'])
    model.state_variable_range['I'] = np.array(param_zerlaut['initial_condition']['I'])
    if param_zerlaut['order'] == 2:
        model.state_variable_range['C_ee'] = np.array(param_zerlaut['initial_condition']['C_ee'])
        model.state_variable_range['C_ei'] = np.array(param_zerlaut['initial_condition']['C_ei'])
        model.state_variable_range['C_ii'] = np.array(param_zerlaut['initial_condition']['C_ii'])
    model.state_variable_range['W_e'] = np.array(param_zerlaut['initial_condition']['W_e'])
    model.state_variable_range['W_i'] = np.array(param_zerlaut['initial_condition']['W_i'])

    ##Solution equation
    def equation(x):
        return (model.TF_inhibitory(fezero, x,0.,0.,0.)-x)
        #return (model.TF_inhibitory((fezero+extinp), x,fezero*model.b_e*model.tau_w_e)-x) # with fix adaptation

    # raneg fo frequency in KHz
    xrange1 = np.arange(0.0, 200.0, 0.001)*1e-3
    feprimovec=np.zeros_like(xrange1)
    fiprimovec=np.zeros_like(xrange1)
    # print(len(xrange1))
    init_fi=0.0
    for i in range(len(xrange1)):
        # print(i)
        fezero=xrange1[i]
        fizero=fsolve(equation, init_fi,xtol=1.e-35)
        while fizero < 0.0:
            init_fi+=0.0001
            fizero=fsolve(equation, init_fi,xtol=1.e-35)
        init_fi=np.abs(fizero)
        fezeroprime=model.TF_excitatory(fezero, fizero,0.,0.,0.)
        #fezeroprime=model.TF_excitatory(fezero, fizero,fezero*model.b_e*model.tau_w_e) # with fixe adaptation
        feprimovec[i]=fezeroprime
        fiprimovec[i]=fizero
        # print(i,np.array(xrange1[i]),fezeroprime,fizero)
    import matplotlib.pyplot as plt
    plt.plot(xrange1,feprimovec,'k.',xrange1,xrange1,'k--')
    plt.plot(fiprimovec,feprimovec,'r.')
    plt.plot(xrange1,fiprimovec,'b.')
    plt.plot(xrange1,fiprimovec-feprimovec,'g.')
    plt.show()