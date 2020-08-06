import nest
import numpy as np
import matplotlib.pylab as plt
import sys


def slidding_window(data,width):
    res = np.zeros((data.shape[0]-width,width))
    res [:,:] = data[[[ i+j for i in range(width) ] for j in range(data.shape[0]-width)]]
    return res.mean(axis=1)

def slidding_window_co(data_1,data_2,width):
    res_1 = np.zeros((data_1.shape[0]-width,width))
    res_1 [:,:] = data_1[[[ i+j for i in range(width) ] for j in range(data_1.shape[0]-width)]]
    res_2 = np.zeros((data_2.shape[0]-width,width))
    res_2 [:,:] = data_2[[[ i+j for i in range(width) ] for j in range(data_2.shape[0]-width)]]
    return np.cov(res_1,res_2)


def compute_spike_count(data,time):
    """
    Compute the firing rate
    :param data: the spike of all neurons between end and begin
    :return: the mean and the standard deviation of firing rate, the maximum and minimum of firing rate
    """
    #get data
    n_fil = np.searchsorted(time,data)
    n_fil = n_fil.astype(int)
    #count the number of the same id
    count_of_t = np.bincount(n_fil)
    #compute the rate
    rate_each_t_incomplet = count_of_t
    rate_each_t = np.concatenate(
        (rate_each_t_incomplet, np.zeros(len(time)-np.shape(rate_each_t_incomplet)[0] )))
    return rate_each_t

def compute_rate(data,time,DT,N):
    """
    Compute the firing rate
    :param data: the spike of all neurons between end and begin
    :return: the mean and the standard deviation of firing rate, the maximum and minimum of firing rate
    """
    #get data
    n_fil = np.searchsorted(time,data)
    n_fil = n_fil.astype(int)
    #count the number of the same id
    count_of_t = np.bincount(n_fil)/float(N)
    #compute the rate
    rate_each_t_incomplet = count_of_t
    rate_each_t = np.concatenate(
        (rate_each_t_incomplet, np.zeros(len(time)-np.shape(rate_each_t_incomplet)[0] )))
    return rate_each_t/DT


def bin_array_co(data_ex,data_in, time_array,DT,N,width):
    hist_ex = compute_rate(data_ex,time_array,DT*1e-3,N)
    hist_in = compute_rate(data_in,time_array,DT*1e-3,N)
    slide_data_ex = slidding_window(hist_ex,width)
    slide_data_in = slidding_window(hist_in,width)
    co_data = slidding_window_co(hist_ex,hist_in,width)
    slide_time = slidding_window(time_array,width)
    return  (
            slide_time,
            slide_data_ex,
            slide_data_in,
            co_data
            )
def bin_array(array, BIN, time_array,DT,N,width):
    hist = compute_rate(array,time_array,DT*1e-3,N)
    slide_data = slidding_window(hist,width)
    slide_time = slidding_window(time_array,width)
    N0 = int(BIN/(time_array[1]-time_array[0]))
    N1 = int((time_array[-1]-time_array[0])/BIN)
    return (time_array[:N0*N1].reshape((N1,N0)).mean(axis=1),
            hist[:N0*N1].reshape((N1,N0)).mean(axis=1),
            slide_time,
            slide_data
            )
def run(b,path,connection_bernoulli,noise_pop):
    N1 = 2000
    N2 = 8000
    TotTime=3.e3
    time_release = 1000.0
    # TotTime=1.e3
    # time_release = 0.0

    DT=0.1
    BIN=5
    T = int(20/DT)
    nest.SetKernelStatus({'resolution':DT,
                          "total_num_virtual_procs":10,
                          'grng_seed':10,
                          'print_time':True})
    
    model = 'aeif_cond_exp'
    
    G1 = nest.Create(model,N1)
    nest.SetStatus(G1,{
        'b':0.0,
        'V_peak':0.0,
        'V_reset':-65.0,
        't_ref':5.0,
        'V_m': -65.0,
        'w':0.0,
        'g_ex':0.0,
        'g_in':0.0,
        'C_m':200.0,
        'g_L':10.0,
        'I_e':0.0,
        'a':0.0,
        'E_L':-65.0,
        'V_th':-50.0,
        'Delta_T':0.5,
        'tau_w':1.0,
        'E_ex':0.,
        'E_in':-80.,
        'tau_syn_ex':5.0,
        'tau_syn_in':5.0,
    })
    
    G2 = nest.Create(model,N2)
    nest.SetStatus(G2,{
        'b':b,
        'V_peak':0.0,
        'V_reset':-64.5,
        't_ref':5.0,
        'V_m': -64.5,
        'w':0.0,
        'g_ex':0.0,
        'g_in':0.0,
        'C_m':200.0,
        'g_L':10.0,
        'I_e':0.0,
        'a':0.0,
        'E_L':-64.5,
        'V_th':-50.0,
        'Delta_T':2.0,
        'tau_w':500.0,
        'E_ex':0.,
        'E_in':-80.,
        'tau_syn_ex':5.0,
        'tau_syn_in':5.0,
    })
    Qi=3.5 # be careful I change the parameter
    Qe=1.
    
    prbC= 0.05
    prbC2=0.05
    if connection_bernoulli:
        conn_dict1 = {'rule': 'pairwise_bernoulli', 'p': prbC }
        conn_dict2 = {'rule': 'pairwise_bernoulli', 'p': prbC }
    else:
        # bad connectivity
        conn_dict1 = {'rule': 'fixed_indegree', 'indegree':int(prbC*N1)}
        conn_dict2 = {'rule': 'fixed_indegree', 'indegree':int(prbC2*N2)}

    nest.CopyModel("static_synapse","inhibitory",{"weight":-Qi, "delay":nest.GetKernelStatus("min_delay")})
    nest.CopyModel("static_synapse","excitatory",{"weight":Qe, "delay":nest.GetKernelStatus("min_delay")})
    nest.Connect(G1,G1,conn_dict1,syn_spec="inhibitory")
    nest.Connect(G1,G2,conn_dict1,syn_spec="inhibitory")
    nest.Connect(G2,G1,conn_dict2,syn_spec="excitatory")
    nest.Connect(G2,G2,conn_dict2,syn_spec="excitatory")
    rate = 300.0
    if noise_pop:
        P_ed_2 = nest.Create('poisson_generator',N2,params={'rate':rate/int(prbC2*N2)})
        parrot_2 = nest.Create("parrot_neuron",N2)
        nest.Connect(P_ed_2,parrot_2,'one_to_one',syn_spec="excitatory")
        nest.Connect(parrot_2,G1,conn_dict2,syn_spec="excitatory")
        nest.Connect(parrot_2,G2,conn_dict2,syn_spec="excitatory")
    else:
        P_ed_1 = nest.Create('poisson_generator',1,params={'rate':rate})
        P_ed_2 = nest.Create('poisson_generator',1,params={'rate':rate})
        parrot_1 = nest.Create("parrot_neuron",N1)
        parrot_2 = nest.Create("parrot_neuron",N2)
        nest.Connect(P_ed_1,parrot_1,'all_to_all',syn_spec="excitatory")
        nest.Connect(P_ed_2,parrot_2,'all_to_all',syn_spec="excitatory")
        nest.Connect(parrot_1,G1,'one_to_one',syn_spec="excitatory")
        nest.Connect(parrot_2,G2,'one_to_one',syn_spec="excitatory")



    M1G1 = nest.Create('spike_detector',params={'start':time_release})
    nest.Connect(G1,M1G1)
    M1G2 = nest.Create('spike_detector',params={'start':time_release})
    nest.Connect(G2,M1G2)
    M1PD = nest.Create('spike_detector',params={'start':time_release})
    if not noise_pop:
        nest.Connect(parrot_1,M1PD)
    nest.Connect(parrot_2,M1PD)
    interval = 0.1
    M2G2 = nest.Create('multimeter',1,params={'record_from': ['w'],'start':time_release,'interval':interval})
    nest.Connect(M2G2,G2)
    
    nest.Simulate(time_release+TotTime)
    
    print("compute the voltage\n")
    data =nest.GetStatus(M2G2)[0]['events']
    time,reverse = np.unique(data['times'],return_inverse=True)
    LwG2 = np.zeros((N2,time.size))
    LwG2[data['senders']-N1-1,reverse] = data['w']
    mean_LwG2 = np.mean(LwG2,axis=0)
    del LwG2
    
    print('compute the rate')
    time_array = np.arange(DT,DT*(time.size+2),DT)
    timeRG1,popRateG1,slidetimeRG1,slideRG1=bin_array(nest.GetStatus(M1G1)[0]['events']['times']-time_release,BIN, time_array,DT,N1,T)
    
    timeRG2,popRateG2,slidetimeRG2,slideRG2=bin_array(nest.GetStatus(M1G2)[0]['events']['times']-time_release,BIN, time_array,DT,N2,T)
    if noise_pop:
        timeRPG,popPGRateG,slidetimeRPG,slideRPG=bin_array(nest.GetStatus(M1PD)[0]['events']['times']-time_release,BIN, time_array,DT,N2,T)
    else:
        timeRPG,popPGRateG,slidetimeRPG,slideRPG=bin_array(nest.GetStatus(M1PD)[0]['events']['times']-time_release,BIN, time_array,DT,N2+N1,T)
    
    np.save(path+"/network_b_"+str(b)+".npy",[
                                 time_array,slideRG1,slideRG2,
                                 slideRPG,
                                 time,mean_LwG2])
    
    print(popRateG1.shape)
    print(np.std(popRateG1[100:])/np.mean(popRateG1[100:]))
    print(np.std(popRateG2[100:])/np.mean(popRateG2[100:]))
    plt.plot(slideRG1,label="inhibitory")
    plt.plot(slideRG2,label="excitatory")
    plt.plot(mean_LwG2)
    if noise_pop:
        plt.plot(slideRPG,label="PG excitatory")
    else:
        plt.plot(slideRPG/(prbC2*N2), label="PG excitatory")
    plt.legend()
    plt.savefig(path+"/network_b_"+str(b)+".png")
    plt.close("all")
    nest.ResetKernel()

def run_g(path,pr,rate):
    N1 = 2000
    N2 = 8000
    TotTime=3.e3
    time_release = 1000.0
    # TotTime=1.e3
    # time_release = 0.0

    DT=0.1
    BIN=5
    T = int(20/DT)
    nest.SetKernelStatus({'resolution':DT,
                          "total_num_virtual_procs":14,
                          'grng_seed':10,
                          'print_time':True})

    model = 'aeif_cond_exp'

    G1 = nest.Create(model,N1)
    nest.SetStatus(G1,{
        'b':0.0,
        'V_peak':0.0,
        'V_reset':-65.0,
        't_ref':5.0,
        'V_m': -65.0,
        'w':0.0,
        'g_ex':0.0,
        'g_in':0.0,
        'C_m':200.0,
        'g_L':10.0,
        'I_e':0.0,
        'a':0.0,
        'E_L':-65.0,
        'V_th':-50.0,
        'Delta_T':0.5,
        'tau_w':1.0,
        'E_ex':0.,
        'E_in':-80.,
        'tau_syn_ex':5.0,
        'tau_syn_in':5.0,
    })

    G2 = nest.Create(model,N2)
    nest.SetStatus(G2,{
        'b':pr,
        'V_peak':0.0,
        'V_reset':-67.0,
        't_ref':5.0,
        'V_m': -67.0,
        'w':0.0,
        'g_ex':0.0,
        'g_in':0.0,
        'C_m':200.0,
        'g_L':10.0,
        'I_e':0.0,
        'a':0.0,
        'E_L':-64.5,
        'V_th':-50.0,
        'Delta_T':2.0,
        'tau_w':1000.0,
        'E_ex':0.,
        'E_in':-80.,
        'tau_syn_ex':5.0,
        'tau_syn_in':5.0,
    })
    Qi=6.5
    Qe=1.

    prbC  = 0.10
    prbC_in = 0.05
    conn_dict = {'rule': 'pairwise_bernoulli', 'p': prbC }
    conn_dict_in = {'rule': 'pairwise_bernoulli', 'p': prbC_in }
    conn_dict_in_in = {'rule': 'pairwise_bernoulli', 'p': prbC_in }

    nest.CopyModel("static_synapse","inhibitory",{"weight":-Qi, "delay":nest.GetKernelStatus("min_delay")})
    nest.CopyModel("static_synapse","excitatory",{"weight":Qe, "delay":nest.GetKernelStatus("min_delay")})
    nest.Connect(G1,G1,conn_dict_in_in,syn_spec="inhibitory")
    nest.Connect(G1,G2,conn_dict_in,syn_spec="inhibitory")# 0.06
    nest.Connect(G2,G1,conn_dict,syn_spec="excitatory")
    nest.Connect(G2,G2,conn_dict,syn_spec="excitatory")

    P_ed = nest.Create('poisson_generator',1,params={'rate':rate})
    parrot_1 = nest.Create("parrot_neuron",N1)
    parrot_2 = nest.Create("parrot_neuron",N2)
    nest.Connect(P_ed,parrot_1,'all_to_all',syn_spec="excitatory")
    nest.Connect(P_ed,parrot_2,'all_to_all',syn_spec="excitatory")
    nest.Connect(parrot_1,G1,'one_to_one',syn_spec="excitatory")
    nest.Connect(parrot_2,G2,'one_to_one',syn_spec="excitatory")

    M1G1 = nest.Create('spike_detector',params={'start':time_release})
    nest.Connect(G1,M1G1)
    M1G2 = nest.Create('spike_detector',params={'start':time_release})
    nest.Connect(G2,M1G2)
    M1PD = nest.Create('spike_detector',params={'start':time_release})
    nest.Connect(parrot_2,M1PD,'all_to_all')
    interval = 0.1
    M2G2 = nest.Create('multimeter',1,params={'record_from': ['w'],'start':time_release,'interval':interval})
    nest.Connect(M2G2,G2)

    nest.Simulate(time_release+TotTime)

    print("compute the voltage\n")
    data =nest.GetStatus(M2G2)[0]['events']
    time,reverse = np.unique(data['times'],return_inverse=True)
    LwG2 = np.zeros((N2,time.size))
    LwG2[data['senders']-N1-1,reverse] = data['w']
    mean_LwG2 = np.mean(LwG2,axis=0)
    del LwG2

    print('compute the rate')
    time_array = np.arange(DT,DT*(time.size+2),DT)
    timeRG1,popRateG1,slidetimeRG1,slideRG1=bin_array(nest.GetStatus(M1G1)[0]['events']['times']-time_release,BIN, time_array,DT,N1,T)

    timeRG2,popRateG2,slidetimeRG2,slideRG2=bin_array(nest.GetStatus(M1G2)[0]['events']['times']-time_release,BIN, time_array,DT,N2,T)
    timeRPG,popPGRateG,slidetimeRPG,slideRPG=bin_array(nest.GetStatus(M1PD)[0]['events']['times']-time_release,BIN, time_array,DT,N2,T)

    np.save(path+"/network_pr_"+str(pr)+"_rate_"+str(rate)+".npy",[
                                 time_array,slideRG1,slideRG2,
                                 slideRPG,
                                 time,mean_LwG2])

    print(popRateG1.shape)
    print(np.std(popRateG1[100:])/np.mean(popRateG1[100:]))
    print(np.std(popRateG2[100:])/np.mean(popRateG2[100:]))
    plt.figure(figsize=(20,8))
    plt.plot(slideRG1,label="inhibitory")
    plt.plot(slideRG2,label="excitatory")
    plt.plot(mean_LwG2)
    plt.plot(slideRPG/50,label="PG excitatory")
    plt.legend()
    plt.savefig(path+"/network_pr_"+str(pr)+"_rate_"+str(rate)+".png")
    plt.close("all")
    nest.ResetKernel()


# for b in np.arange(0.,11.,1.):
#     run(b, "/home/kusch/Documents/project/co_simulation/co-simulation_mouse/test_nest/test_file/network_sim/bernoulli/", True,False)
# for b in np.arange(0.,11.,1.):
#     run(b, "/home/kusch/Documents/project/co_simulation/co-simulation_mouse/test_nest/test_file/network_sim/noise_pop/", True,True)
for pr in np.arange(1.,20.0,1.0):
    for rate in np.arange(250.0, 300.0, 10.0):
        run_g("/home/kusch/Documents/project/co_simulation/co-simulation-tvb-nest/test_nest/test_file/network_Q_i_adp_2/", pr, rate)
