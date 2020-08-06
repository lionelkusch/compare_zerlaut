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
b=6.0
N1 = 2000
N2 = 8000
TotTime=3.e3
# TotTime=  1000.0
time_release = 1000.0
DT=0.1
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

# P_ed_1 = nest.Create('poisson_generator',N1,params={'rate':0.7+1.e-3})
P_ed_2 = nest.Create('poisson_generator',N2,params={'rate':0.7+1.e-3})
# parrot_1 = nest.Create("parrot_neuron",N1)
parrot_2 = nest.Create("parrot_neuron",N2)

Qi=2.5
Qe=1.

prbC= 0.05
prbC2=0.05
conn_dict1 = {'rule': 'pairwise_bernoulli', 'p': prbC,
              # 'allow_autapses': False, 'allow_multapses': False
              }
# conn_dict1 = {'rule': 'fixed_indegree', 'indegree':int(prbC*N1), 'allow_autapses': False, 'allow_multapses': False}
nest.CopyModel("static_synapse","inhibitory",{"weight":-Qi, "delay":nest.GetKernelStatus("min_delay")})
conn_dict2 = {'rule': 'pairwise_bernoulli', 'p': prbC,
              # 'allow_autapses': False, 'allow_multapses': False
              }
# conn_dict2 = {'rule': 'fixed_indegree', 'indegree':int(prbC2*N2), 'allow_autapses': False, 'allow_multapses': True}
nest.CopyModel("static_synapse","excitatory",{"weight":Qe, "delay":nest.GetKernelStatus("min_delay")})
# nest.CopyModel("static_synapse","poisson",{"weight":50.0, "delay":nest.GetKernelStatus("min_delay")})
nest.Connect(G1,G1,conn_dict1,syn_spec="inhibitory")
nest.Connect(G1,G2,conn_dict1,syn_spec="inhibitory")
nest.Connect(G2,G1,conn_dict2,syn_spec="excitatory")
nest.Connect(G2,G2,conn_dict2,syn_spec="excitatory")
# nest.Connect(P_ed_1,parrot_1,'one_to_one',syn_spec="excitatory")
nest.Connect(P_ed_2,parrot_2,'one_to_one',syn_spec="excitatory")
nest.Connect(parrot_2,G1,conn_dict2,syn_spec="excitatory")
nest.Connect(parrot_2,G2,conn_dict2,syn_spec="excitatory")

M1G1 = nest.Create('spike_detector',params={'start':time_release})
nest.Connect(G1,M1G1)
M1G2 = nest.Create('spike_detector',params={'start':time_release})
nest.Connect(G2,M1G2)
# M1PD1 = nest.Create('spike_detector',params={'start':time_release})
M1PD2 = nest.Create('spike_detector',params={'start':time_release})
# nest.Connect(parrot_1,M1PD1)
nest.Connect(parrot_2,M1PD2)
interval = 0.1
# M2G1 = nest.Create('multimeter',1,params={'record_from': ['V_m', 'w'],'start':time_release,'interval':interval})
# nest.Connect(M2G1,G1)
# M2G2 = nest.Create('multimeter',1,params={'record_from': ['V_m', 'w'],'start':time_release,'interval':interval})
M2G2 = nest.Create('multimeter',1,params={'record_from': ['w'],'start':time_release,'interval':interval})
nest.Connect(M2G2,G2)

nest.Simulate(time_release+TotTime)

print("compute the voltage\n")
# Lt1G1 = np.arange(interval,TotTime+interval,interval)
Lt1G2 = np.arange(interval,TotTime+interval,interval)

# LVG1 = np.zeros((N1,len(Lt1G1)))
# LwG1 = np.zeros((N1,len(Lt1G1)))
# data =nest.GetStatus(M2G1)[0]['events']
# time = np.ceil((data['times'] -time_release)/ interval-1).astype(np.int)
# LVG1[data['senders']-1,time] = data['V_m']
# LwG1[data['senders']-1,time] = data['w']
data =nest.GetStatus(M2G2)[0]['events']
time,reverse = np.unique(data['times'],return_inverse=True)
# LVG2 = np.zeros((N2,len(Lt1G2)))
LwG2 = np.zeros((N2,time.size))
# LVG2[data['senders']-N1-1,time] = data['V_m']
LwG2[data['senders']-N1-1,reverse] = data['w']


# mean_LVG1 = np.mean(LVG1,axis=0)
# max_LVG1 = np.max(LVG1,axis=0)
# min_LVG1 = np.min(LVG1,axis=0)
# mean_LwG1 = np.mean(LwG1,axis=0)
# max_LwG1 = np.max(LwG1,axis=0)
# min_LwG1 = np.min(LwG1,axis=0)
# mean_LVG2 = np.mean(LVG2,axis=0)
# max_LVG2 = np.max(LVG2,axis=0)
# min_LVG2 = np.min(LVG2,axis=0)
mean_LwG2 = np.mean(LwG2,axis=0)
del LwG2
# max_LwG2 = np.max(LwG2,axis=0)
# min_LwG2 = np.min(LwG2,axis=0)

print('compute the rate')
BIN=5
T = int(20/DT)
time_array = np.arange(DT,DT*(time.size+2),DT)
timeRG1,popRateG1,slidetimeRG1,slideRG1=bin_array(nest.GetStatus(M1G1)[0]['events']['times']-time_release,BIN, time_array,DT,N1,T)
# timeRPG1,popPGRateG1,slidetimeRPG1,slideRPG1=bin_array(nest.GetStatus(M1PD1)[0]['events']['times']-time_release,BIN, time_array,DT,N1,T)

timeRG2,popRateG2,slidetimeRG2,slideRG2=bin_array(nest.GetStatus(M1G2)[0]['events']['times']-time_release,BIN, time_array,DT,N2,T)
timeRPG2,popPGRateG2,slidetimeRPG2,slideRPG2=bin_array(nest.GetStatus(M1PD2)[0]['events']['times']-time_release,BIN, time_array,DT,N2,T)

# timeRM,popMeanPD,slidetimeRM,slideRM=bin_array(np.concatenate([nest.GetStatus(M1PD2)[0]['events']['times'],
#                                               nest.GetStatus(M1PD1)[0]['events']['times']])-time_release,BIN, time_array,DT,N1+N2,T)

# print('Plot')
#
# fig=plt.figure(figsize=(12,4))
# ax1=fig.add_subplot(231)
# ax2=fig.add_subplot(232)
#
# for a in range(1):
#     ax1.plot(Lt1G1, LVG1[a],'r:',linewidth=0.5)
#     ax1.plot(Lt1G2, LVG2[a],'g:',linewidth=0.5)
#
# for a in range(5):
#     ax2.plot(Lt1G1, LwG1[a],'r:',linewidth=0.5)
#     ax2.plot(Lt1G2, LwG2[a],'g:',linewidth=0.5)
#
# ax1.plot(Lt1G1, mean_LVG1,'r',linewidth=2.0)
# ax2.plot(Lt1G1, mean_LwG1,'r',linewidth=2.0)
# ax1.plot(Lt1G2, mean_LVG2,'g',linewidth=2.0)
# ax2.plot(Lt1G2, mean_LwG2,'g',linewidth=2.0)
# # ax1.plot(Lt1G1, max_LVG1,'r--',linewidth=0.5)
# ax2.plot(Lt1G1, max_LwG1,'r--',linewidth=1.0)
# # ax1.plot(Lt1G2, max_LVG2,'g--',linewidth=0.5)
# ax2.plot(Lt1G2, max_LwG2,'g--',linewidth=1.0)
# ax1.plot(Lt1G1, min_LVG1,'r--',linewidth=0.5)
# ax2.plot(Lt1G1, min_LwG1,'r--',linewidth=1.0)
# ax1.plot(Lt1G2, min_LVG2,'g--',linewidth=0.5)
# ax2.plot(Lt1G2, min_LwG2,'g--',linewidth=1.0)
#
#
# ax1.set_ylim([-100, 0])
# ax1.set_xlabel('Time (ms)')
# ax1.set_ylabel('V in (mV)')
# ax2.set_xlabel('Time (ms)')
# ax2.set_ylabel('W in (pA)')
#
# ax3=fig.add_subplot(233)
# ax3.plot(nest.GetStatus(M1G1)[0]['events']['times'], nest.GetStatus(M1G1)[0]['events']['senders'], '.r',markersize=0.1)
# ax3.plot(nest.GetStatus(M1G2)[0]['events']['times'], nest.GetStatus(M1G2)[0]['events']['senders'], '.g',markersize=0.1)
# ax3.set_xlabel('Time (ms)')
# ax3.set_ylabel('Neuron index')
#
# ax4=fig.add_subplot(234)
# ax4.plot(slidetimeRG1,slideRG1, 'r')
# ax4.plot(slidetimeRG2,slideRG2, 'g')
# ax4.set_xlabel('Time (ms)')
# ax4.set_ylabel('FR')
# print('slide rate')
#
# ax5=fig.add_subplot(235)
# ax5.plot(slidetimeRPG1,slideRPG1-slideRPG2, 'g')
# ax5.plot(slidetimeRM,slideRM, 'r')
# ax5.set_xlabel('Time (ms)')
# ax5.set_ylabel('FR')
# print('slide mean rate')
#
# ax6=fig.add_subplot(236)
# ax6.plot(slidetimeRPG2,slideRPG2, 'g')
# ax6.plot(slidetimeRPG1,slideRPG1, 'r')
# ax6.set_xlabel('Time (ms)')
# ax6.set_ylabel('FR')
# print('slide PG mean rate')
#
#
# fig=plt.figure(figsize=(12,4))
# ax1=fig.add_subplot(231)
# ax2=fig.add_subplot(232)
#
# for a in range(1):
#     ax1.plot(Lt1G1, LVG1[a],'r:',linewidth=0.5)
#     ax1.plot(Lt1G2, LVG2[a],'g:',linewidth=0.5)
#
# for a in range(5):
#     ax2.plot(Lt1G1, LwG1[a],'r:',linewidth=0.5)
#     ax2.plot(Lt1G2, LwG2[a],'g:',linewidth=0.5)
#
# ax1.plot(Lt1G1, mean_LVG1,'r',linewidth=2.0)
# ax2.plot(Lt1G1, mean_LwG1,'r',linewidth=2.0)
# ax1.plot(Lt1G2, mean_LVG2,'g',linewidth=2.0)
# ax2.plot(Lt1G2, mean_LwG2,'g',linewidth=2.0)
# # ax1.plot(Lt1G1, max_LVG1,'r--',linewidth=0.5)
# ax2.plot(Lt1G1, max_LwG1,'r--',linewidth=1.0)
# # ax1.plot(Lt1G2, max_LVG2,'g--',linewidth=0.5)
# ax2.plot(Lt1G2, max_LwG2,'g--',linewidth=1.0)
# ax1.plot(Lt1G1, min_LVG1,'r--',linewidth=0.5)
# ax2.plot(Lt1G1, min_LwG1,'r--',linewidth=1.0)
# ax1.plot(Lt1G2, min_LVG2,'g--',linewidth=0.5)
# ax2.plot(Lt1G2, min_LwG2,'g--',linewidth=1.0)
#
#
# ax1.set_ylim([-100, 0])
# ax1.set_xlabel('Time (ms)')
# ax1.set_ylabel('V in (mV)')
# ax2.set_xlabel('Time (ms)')
# ax2.set_ylabel('W in (pA)')
#
# ax3=fig.add_subplot(233)
# ax3.plot(nest.GetStatus(M1G1)[0]['events']['times'], nest.GetStatus(M1G1)[0]['events']['senders'], '.r',markersize=0.1)
# ax3.plot(nest.GetStatus(M1G2)[0]['events']['times'], nest.GetStatus(M1G2)[0]['events']['senders'], '.g',markersize=0.1)
# ax3.set_xlabel('Time (ms)')
# ax3.set_ylabel('Neuron index')
#
# ax4=fig.add_subplot(234)
# ax4.plot(timeRG1,popRateG1, 'r')
# ax4.plot(timeRG2,popRateG2, 'g')
# ax4.set_xlabel('Time (ms)')
# ax4.set_ylabel('FR')
#
# ax5=fig.add_subplot(235)
# ax5.plot(timeRM,popPGRateG2-popPGRateG1, 'g')
# ax5.plot(timeRM,popMeanPD, 'r')
# ax5.set_xlabel('Time (ms)')
# ax5.set_ylabel('FR')
#
# ax6=fig.add_subplot(236)
# ax6.plot(timeRPG2,popPGRateG2, 'g')
# ax6.plot(timeRPG1,popPGRateG1, 'r')
# ax6.set_xlabel('Time (ms)')
# ax6.set_ylabel('FR')

np.save("./network_b_"+str(b)+".npy",[
                            # popRateG1,popRateG2,popPGRateG1,popPGRateG2,popMeanPD,
                             slideRG1,slideRG2,
                             slideRPG2,
                             # slideRPG1,slideRPG2,slideRM,
                             mean_LwG2])

print(popRateG1.shape)
print(np.std(popRateG1[100:])/np.mean(popRateG1[100:]))
print(np.std(popRateG2[100:])/np.mean(popRateG2[100:]))
# print(np.std(popMeanPD[100:])/np.mean(popMeanPD[100:]))
plt.plot(slideRG1,label="inhibitory")
plt.plot(slideRG2,label="excitatory")
# plt.plot(slideRM,label="mean")
# plt.plot(slideRPG1,label="PG inhibitory ")
plt.plot(slideRPG2*400.0,label="PG excitatory")
plt.plot(mean_LwG2)
plt.legend()
plt.show()