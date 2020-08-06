import nest
import numpy as np
import matplotlib.pylab as plt
import itertools
import sys

nest.set_verbosity(100)


def compute_rate(data,time,N,Dt):
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
    rate_each_t_incomplet = count_of_t / float(N)
    rate_each_t = np.concatenate(
        (rate_each_t_incomplet, np.zeros(len(time)-np.shape(rate_each_t_incomplet)[0] )))
    return rate_each_t/(Dt*1e-3)

def bin_array(array, BIN, time_array,N,Dt):
    hist = compute_rate(array,time_array,N,Dt)
    N0 = int(BIN/(time_array[1]-time_array[0]))
    N1 = int((time_array[-1]-time_array[0])/BIN)
    return (time_array[:N0*N1].reshape((N1,N0)).mean(axis=1),
            hist[:N0*N1].reshape((N1,N0)).mean(axis=1))

def run(b,rate):
    N1 = 2000
    N2 = 8000
    TotTime=  2000.0
    time_kick = 1000.0
    time_relase= 10000.0
    time_start =  time_kick+time_relase
    DT = 0.1

    nest.SetKernelStatus({'resolution': DT,
                          "total_num_virtual_procs": 15,
                          'grng_seed': 16,
                          'print_time': True})

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
    P_ed_1_kick = nest.Create('poisson_generator', params={'rate':340.-rate,'start':0.0,'stop':time_kick})
    P_ed_2_kick = nest.Create('poisson_generator', params={'rate':340.-rate,'start':0.0,'stop':time_kick})
    P_ed_1 = nest.Create('poisson_generator', params={'rate':rate})
    P_ed_2 = nest.Create('poisson_generator', params={'rate':rate})
    
    Qi=3.0
    Qe=1.
    
    prbC= 0.05
    prbC2=0.05
    conn_dict1 = {'rule': 'pairwise_bernoulli', 'p': prbC}#, 'allow_autapses': False, 'allow_multapses': False}
    # conn_dict1 = {'rule': 'fixed_indegree', 'indegree':int(prbC*N1), 'allow_autapses': False, 'allow_multapses': False}
    nest.CopyModel("static_synapse","inhibitory",{"weight":-Qi, "delay":nest.GetKernelStatus("min_delay")})
    conn_dict2 = {'rule': 'pairwise_bernoulli', 'p': prbC2,}# 'allow_autapses': False, 'allow_multapses': False}
    # conn_dict2 = {'rule': 'fixed_indegree', 'indegree':int(prbC2*N2), 'allow_autapses': False, 'allow_multapses': True}
    nest.CopyModel("static_synapse","excitatory",{"weight":Qe, "delay":nest.GetKernelStatus("min_delay")})
    nest.CopyModel("static_synapse","poisson_kick",{"weight":Qe, "delay":nest.GetKernelStatus("min_delay")})
    nest.CopyModel("static_synapse","poisson",{"weight":Qe, "delay":nest.GetKernelStatus("min_delay")})
    nest.Connect(G1,G1,conn_dict1,syn_spec="inhibitory")
    nest.Connect(G1,G2,conn_dict1,syn_spec="inhibitory")
    nest.Connect(G2,G1,conn_dict2,syn_spec="excitatory")
    nest.Connect(G2,G2,conn_dict2,syn_spec="excitatory")
    nest.Connect(P_ed_1_kick,G1,'all_to_all',syn_spec="poisson_kick")
    nest.Connect(P_ed_2_kick,G2,'all_to_all',syn_spec="poisson_kick")
    nest.Connect(P_ed_1,G1,'all_to_all',syn_spec="poisson")
    nest.Connect(P_ed_2,G2,'all_to_all',syn_spec="poisson")
    
    M1G1 = nest.Create('spike_detector', params={'start':time_start})
    nest.Connect(G1,M1G1)
    M1G2 = nest.Create('spike_detector', params={'start':time_start})
    nest.Connect(G2,M1G2)
    interval = 1.0
    M2G1 = nest.Create('multimeter', params={'record_from': ['V_m', 'w'],'interval':interval,'start':time_start})
    nest.Connect(M2G1,G1)
    M2G2 = nest.Create('multimeter', params={'record_from': ['V_m', 'w'],'interval':interval,'start':time_start})
    nest.Connect(M2G2,G2)
    
    nest.Simulate(TotTime+time_start)

    Lt1G1 = np.arange(interval,TotTime,interval)
    Lt1G2 = np.arange(interval,TotTime,interval)
    
    LVG1 = np.zeros((N1,len(Lt1G1)))
    LwG1 = np.zeros((N1,len(Lt1G1)))
    data =nest.GetStatus(M2G1)[0]['events']
    time = np.ceil(data['times'] / interval - time_start-1).astype(np.int)
    LVG1[data['senders']-1,time] = data['V_m']
    LwG1[data['senders']-1,time] = data['w']
    LVG2 = np.zeros((N2,len(Lt1G2)))
    LwG2 = np.zeros((N2,len(Lt1G2)))
    data =nest.GetStatus(M2G2)[0]['events']
    time = np.ceil(data['times'] / interval - time_start-1).astype(np.int)
    LVG2[data['senders']-N1-1,time] = data['V_m']
    LwG2[data['senders']-N1-1,time] = data['w']
    
    
    mean_LVG1 = np.mean(LVG1,axis=0)
    max_LVG1 = np.max(LVG1,axis=0)
    min_LVG1 = np.min(LVG1,axis=0)
    mean_LwG1 = np.mean(LwG1,axis=0)
    max_LwG1 = np.max(LwG1,axis=0)
    min_LwG1 = np.min(LwG1,axis=0)
    mean_LVG2 = np.mean(LVG2,axis=0)
    max_LVG2 = np.max(LVG2,axis=0)
    min_LVG2 = np.min(LVG2,axis=0)
    mean_LwG2 = np.mean(LwG2,axis=0)
    max_LwG2 = np.max(LwG2,axis=0)
    min_LwG2 = np.min(LwG2,axis=0)
    
    BIN=20
    time_array = np.arange(0,TotTime+1+DT,DT)
    TimBinned,popRateG1=bin_array(nest.GetStatus(M1G1)[0]['events']['times'] - time_start, BIN, time_array, N1, DT)
    
    TimBinned,popRateG2=bin_array(nest.GetStatus(M1G2)[0]['events']['times'] - time_start, BIN, time_array, N2, DT)
    
    

    fig=plt.figure(figsize=(20,8))
    ax1=fig.add_subplot(221)
    ax2=fig.add_subplot(222)
    
    for a in range(1):
        ax1.plot(Lt1G1, LVG1[a],'r:',linewidth=0.5)
        ax1.plot(Lt1G2, LVG2[a],'g:',linewidth=0.5)
    
    for a in range(5):
        ax2.plot(Lt1G1, LwG1[a],'r:',linewidth=0.5)
        ax2.plot(Lt1G2, LwG2[a],'g:',linewidth=0.5)
    
    ax1.plot(Lt1G1, mean_LVG1,'r',linewidth=2.0)
    ax2.plot(Lt1G1, mean_LwG1,'r',linewidth=2.0)
    ax1.plot(Lt1G2, mean_LVG2,'g',linewidth=2.0)
    ax2.plot(Lt1G2, mean_LwG2,'g',linewidth=2.0)
    # ax1.plot(Lt1G1, max_LVG1,'r--',linewidth=0.5)
    ax2.plot(Lt1G1, max_LwG1,'r--',linewidth=1.0)
    # ax1.plot(Lt1G2, max_LVG2,'g--',linewidth=0.5)
    ax2.plot(Lt1G2, max_LwG2,'g--',linewidth=1.0)
    ax1.plot(Lt1G1, min_LVG1,'r--',linewidth=0.5)
    ax2.plot(Lt1G1, min_LwG1,'r--',linewidth=1.0)
    ax1.plot(Lt1G2, min_LVG2,'g--',linewidth=0.5)
    ax2.plot(Lt1G2, min_LwG2,'g--',linewidth=1.0)
    
    
    ax1.set_ylim([-100, 0])
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('V in (mV)')
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('W in (pA)')
    
    ax3=fig.add_subplot(223)
    ax3.plot(nest.GetStatus(M1G1)[0]['events']['times'] - time_start, nest.GetStatus(M1G1)[0]['events']['senders'], '.r', markersize=0.1)
    ax3.plot(nest.GetStatus(M1G2)[0]['events']['times'] - time_start, nest.GetStatus(M1G2)[0]['events']['senders'], '.g', markersize=0.1)
    ax3.set_xlabel('Time (ms)')
    ax3.set_ylabel('Neuron index')
    
    ax4=fig.add_subplot(224)
    ax4.plot(TimBinned,popRateG1, 'r')
    ax4.plot(TimBinned,popRateG2, 'g')
    ax4.set_xlabel('Time (ms)')
    ax4.set_ylabel('FR')
    
    print("population inhibitory",np.mean(popRateG1),np.std(popRateG1),"population excitatory",np.mean(popRateG2),np.std(popRateG2),"adaptation",np.mean(mean_LwG2),np.std(mean_LwG2),"good:",popRateG1[-1]<1.0)
    plt.savefig("/home/kusch/Documents/project/co_simulation/co-simulation_mouse/test_nest/test_file/fitting/b"+str(b)+"rate"+str(rate)+".png")
    # plt.show()
    result = np.array([np.mean(popRateG1),np.std(popRateG1),
                           np.mean(popRateG2),np.std(popRateG2),
                           np.mean(mean_LwG2),np.std(mean_LwG2),
                           popRateG1[-1]<1.0])
    nest.ResetKernel()
    return result
for n in range(10):
    range_rate = np.arange(0.0,0.1,0.1)
    range_b =  np.arange(0.1,1.,0.1)
    b,rate = list(itertools.product(range_b,range_rate))[n]
    print(b,rate)
    result = run(b,0.0)
    np.save("/home/kusch/Documents/project/co_simulation/co-simulation_mouse/test_nest/test_file/fitting/network_"+str(n)+".npy",result)


# if __name__ == "__main__":
#     n = int(sys.argv[1])
#     range_rate = np.arange(0.0,105.0,5.0)
#     range_b =  np.arange(0.,3.1,0.1)
#     b,rate = list(itertools.product(range_b,range_rate))[n]
#     print(b,rate)
#     result = run(b,rate)
#     np.save("/home/kusch/Documents/project/co_simulation/co-simulation_mouse/test_nest/test_file/fitting/network_"+str(n)+".npy",result)
