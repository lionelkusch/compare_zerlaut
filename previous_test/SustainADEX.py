import matplotlib.pyplot as plt
import numpy as np
from brian2 import *

def visualise_connectivity(S):
    Ns = len(S.source)
    Nt = len(S.target)
    figure(figsize=(10, 4))
    subplot(121)
    plot(zeros(Ns), arange(Ns), 'ok', ms=1)
    plot(ones(Nt), arange(Nt), 'ok', ms=1)
    for i, j in zip(S.i, S.j):
        plot([0, 1], [i, j], '-k',linewidth=0.1)
    xticks([0, 1], ['Source', 'Target'])
    ylabel('Neuron index')
    xlim(-0.1, 1.1)
    ylim(-1, max(Ns, Nt))
    subplot(122)
    plot(S.i, S.j, 'ok', ms=1)
    xlim(-1, Ns)
    ylim(-1, Nt)
    xlabel('Source neuron index')
    ylabel('Target neuron index')

def bin_array(array, BIN, time_array):
    N0 = int(BIN/(time_array[1]-time_array[0]))
    N1 = int((time_array[-1]-time_array[0])/BIN)
    return array[:N0*N1].reshape((N1,N0)).mean(axis=1)


Nsim='1'


start_scope()
# parameters
DT=0.1
defaultclock.dt = DT*ms
N1 = 2000
N2 = 8000
TotTime = 1e3
# TotTime=10
duration = TotTime*ms
seed(50)

eqs='''
dv/dt = (-GsynE*(v-Ee)-GsynI*(v-Ei)-gl*(v-El)+ gl*Dt*exp((v-Vt)/Dt)-w + Is)/Cm : volt (unless refractory)
dw/dt = (a*(v-El)-w)/tau_w:ampere
dGsynI/dt = -GsynI/Tsyn : siemens
dGsynE/dt = -GsynE/Tsyn : siemens
Is:ampere
Cm:farad
gl:siemens
El:volt
a:siemens
tau_w:second
Dt:volt
Vt:volt
Ee:volt
Ei:volt
Tsyn:second
'''#% neuron_params


# Population 1 - FS - inhibitory
b1 = 0.0*pA
G1 = NeuronGroup(N1, eqs, threshold='v > 0.0*mV', reset='v = -65*mV', refractory='5*ms', method='heun')
#init:
G1.v = -65.*mV
G1.w = 0.0*pA
G1.GsynI=0.0*nS
G1.GsynE=0.0*nS
#parameters
G1.Cm = 200.*pF
G1.gl = 10.*nS
G1.Is = 0.0
G1.a = 0.0*nS
G1.El = -65.0*mV
G1.Vt = -50.*mV
G1.Dt = 0.5*mV
G1.tau_w = 1.0*ms

G1.Ee=0.*mV
G1.Ei=-80.*mV
G1.Tsyn=5.*ms

# Population 2 - RS - excitatory
b2 = 10.*pA
G2 = NeuronGroup(N2, eqs, threshold='v > 0.0*mV', reset='v = -64.5*mV; w += b2', refractory='5*ms',  method='heun')
G2.Cm = 200.*pF
G2.El = -64.5*mV
G2.gl = 10.*nS
G2.Is = 0.0*nA
G2.a = 0.*nS
G2.Dt = 2.*mV
G2.tau_w = 500.*ms
G2.Vt = -50.*mV
G2.v = -64.5*mV
G2.w = 0.0*pA
G2.GsynI=0.0*nS
G2.GsynE=0.0*nS

G2.Ee=0.*mV
G2.Ei=-80.*mV
G2.Tsyn=5.*ms


# external drive--------------------------------------------------------------------------
rate =140.0*2+400*1e-3
nb_P = 10
P_ed_1 = PoissonGroup(N1*nb_P, (rate/nb_P)*Hz)
P_ed_2 = PoissonGroup(N2*nb_P, (rate/nb_P)*Hz)

# connections-----------------------------------------------------------------------------

print("connection")
Qi=2.5*nS
Qe=1.*nS

prbC= 0.05
prbC2=0.05
S_12 = Synapses(G1, G2, on_pre='GsynI_post+=Qi')
S_12.connect(p=prbC2)

S_11 = Synapses(G1, G1, on_pre='GsynI_post+=Qi')
S_11.connect(condition='i != j',p=prbC2)

S_21 = Synapses(G2, G1, on_pre='GsynE_post+=Qe')
S_21.connect(p=prbC)

S_22 = Synapses(G2, G2, on_pre='GsynE_post+=Qe')
S_22.connect(condition='i != j', p=prbC)

S_ed_in = Synapses(P_ed_1, G1, on_pre='GsynE_post+=Qe')
S_ed_in.connect(condition='j == i % N1')

S_ed_ex = Synapses(P_ed_2, G2, on_pre='GsynE_post+=Qe')
S_ed_ex.connect(condition='j == i % N2')

M1G1 = SpikeMonitor(G1)
M2G1 = StateMonitor(G1, 'v', record=range(N1),dt=1*ms)
M3G1 = StateMonitor(G1, 'w', record=range(N1),dt=1*ms)
FRG1 = PopulationRateMonitor(G1)
#
M1G2 = SpikeMonitor(G2)
M2G2 = StateMonitor(G2, 'v', record=range(N2),dt=1*ms)
M3G2 = StateMonitor(G2, 'w', record=range(N2),dt=1*ms)
FRG2 = PopulationRateMonitor(G2)

print('--##Start simulation##--')
run(duration)
print('--##End simulation##--')


RasG1 = np.array([M1G1.t/ms, [i+N2 for i in M1G1.i]])
RasG2 = np.array([M1G2.t/ms, M1G2.i])

LVG1=[]
LwG1=[]
LVG2=[]
LwG2=[]

for a in range(N1):
    LVG1.append(array(M2G1[a].v/mV))
    LwG1.append(array(M3G1[a].w/pA))
for a in range(N2):
    LVG2.append(array(M2G2[a].v/mV))
    LwG2.append(array(M3G2[a].w/pA))


BIN=5
time_array = np.arange(int(TotTime/DT))*DT

LfrG2=np.array(FRG2.rate/Hz)
TimBinned,popRateG2=bin_array(time_array, BIN, time_array),bin_array(LfrG2, BIN, time_array)

LfrG1=np.array(FRG1.rate/Hz)
TimBinned,popRateG1=bin_array(time_array, BIN, time_array),bin_array(LfrG1, BIN, time_array)

Lt1G1=array(M2G1.t/ms)
Lt2G1=array(M3G1.t/ms)
Lt1G2=array(M2G2.t/ms)
Lt2G2=array(M3G2.t/ms)

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

fig=plt.figure(figsize=(12,4))
ax1=fig.add_subplot(221)
ax2=fig.add_subplot(222)

for a in range(1):
    ax1.plot(Lt1G1, LVG1[a],'r:',linewidth=0.5)
    ax1.plot(Lt1G2, LVG2[a],'g:',linewidth=0.5)

for a in range(5):
    ax2.plot(Lt2G1, LwG1[a],'r:',linewidth=0.5)
    ax2.plot(Lt2G2, LwG2[a],'g:',linewidth=0.5)

ax1.plot(Lt1G1, mean_LVG1,'r',linewidth=2.0)
ax2.plot(Lt2G1, mean_LwG1,'r',linewidth=2.0)
ax1.plot(Lt1G2, mean_LVG2,'g',linewidth=2.0)
ax2.plot(Lt2G2, mean_LwG2,'g',linewidth=2.0)
# ax1.plot(Lt1G1, max_LVG1,'r--',linewidth=0.5)
ax2.plot(Lt2G1, max_LwG1,'r--',linewidth=1.0)
# ax1.plot(Lt1G2, max_LVG2,'g--',linewidth=0.5)
ax2.plot(Lt2G2, max_LwG2,'g--',linewidth=1.0)
ax1.plot(Lt1G1, min_LVG1,'r--',linewidth=0.5)
ax2.plot(Lt2G1, min_LwG1,'r--',linewidth=1.0)
ax1.plot(Lt1G2, min_LVG2,'g--',linewidth=0.5)
ax2.plot(Lt2G2, min_LwG2,'g--',linewidth=1.0)


ax1.set_ylim([-100, 0])
ax1.set_xlabel('Time (ms)')
ax1.set_ylabel('V in (mV)')
ax2.set_xlabel('Time (ms)')
ax2.set_ylabel('W in (pA)')

ax3=fig.add_subplot(223)
ax3.plot(RasG1[0], RasG1[1], '.r',markersize=0.1)
ax3.plot(RasG2[0], RasG2[1], '.g',markersize=0.1)
ax3.set_xlabel('Time (ms)')
ax3.set_ylabel('Neuron index')

ax4=fig.add_subplot(224)
ax4.plot(TimBinned,popRateG1, 'r')
ax4.plot(TimBinned,popRateG2, 'g')
ax4.set_xlabel('Time (ms)')
ax4.set_ylabel('FR')

print(popRateG1.shape)
print(numpy.std(popRateG1[100:])/np.mean(popRateG1[100:]))
print(numpy.std(popRateG2[100:])/np.mean(popRateG2[100:]))
plt.show()
