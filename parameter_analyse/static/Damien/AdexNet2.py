# import libraries
import matplotlib.pyplot as plt
import numpy as np
from brian2 import *

Cnt = 0
Lb = [i * 2 for i in range(60)]
Lei = [i for i in range(100)]
for bVal in Lb:
    for ExIn in Lei:
        Cnt += 1
        #########################################################
        # Define conditions for simulation
        # start Brian scope:
        start_scope()
        # set dt value for integration (ms):
        DT = 0.1
        seed(0)
        defaultclock.dt = DT * ms

        # total duration of the simulation (ms):
        TotTime = 4000
        duration = TotTime * ms

        #######################################################
        # set the number of neuron of each population:
        # inhibitory Fast Spiking (FS, population 1):
        N1 = 2000
        # Excitatory Regular Spiking (RS, population 2):
        N2 = 8000

        ########################################################
        # define equations of the model
        # define units of parameter
        eqs = '''
		dv/dt = (-GsynE*(v-Ee)-GsynI*(v-Ei)-gl*(v-El)+ gl*Dt*exp((v-Vt)/Dt)-w + Is)/Cm : volt (unless refractory)
		dw/dt = (a*(v-El)-w)/tau_w:ampere
		dGsynI/dt = -GsynI/Tsyn : siemens
		dGsynE/dt = -GsynE/Tsyn : siemens
		Pvar:1
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
		'''

        ########################################################
        # Create populations:

        # Population 1 - FS

        b1 = 0.0 * pA  # no adaptation for FS
        # generate the population
        G1 = NeuronGroup(N1, eqs, threshold='v > -47.5*mV', reset='v = -65*mV', refractory='5*ms', method='heun')
        # set values:
        # initial values of variables:
        G1.v = -65 * mV
        G1.w = 0.0 * pA
        G1.GsynI = 0.0 * nS
        G1.GsynE = 0.0 * nS
        # parameters values:
        # soma:
        G1.Cm = 200. * pF
        G1.gl = 10. * nS
        G1.El = -65. * mV
        G1.Vt = -50. * mV
        G1.Dt = 0.5 * mV
        G1.tau_w = 1.0 * ms  # (no adapation, just to do not have error due to zero division)
        G1.a = 0.0 * nS
        G1.Is = 0.0
        # synapses:
        G1.Ee = 0. * mV
        G1.Ei = -80. * mV
        G1.Tsyn = 5. * ms

        # Population 2 - RS
        b2 = bVal * pA
        # generate the population
        G2 = NeuronGroup(N2, eqs, threshold='v > -40.0*mV', reset='v = -55*mV; w += b2', refractory='5*ms',
                         method='heun')
        # set values:
        # initial values of variables:
        G2.v = -65. * mV
        G2.w = 0.0 * pA
        G2.GsynI = 0.0 * nS
        G2.GsynE = 0.0 * nS
        # parameters values:
        # soma:
        G2.Cm = 200. * pF
        G2.gl = 10. * nS
        G2.El = -63. * mV
        G2.Vt = -50. * mV
        G2.Dt = 2. * mV
        G2.tau_w = 500 * ms
        G2.a = 0. * nS
        G2.Is = 0. * nA
        # synpases:
        G2.Ee = 0. * mV
        G2.Ei = -80. * mV
        G2.Tsyn = 5. * ms

        #######################################################
        # external drive---------------------------------------

        P_ed = PoissonGroup(8000, rates=ExIn * Hz)

        #######################################################
        # connections-------------------------------------------
        # quantal increment when spike:
        Qi = 5. * nS
        Qe = 1.5 * nS

        # probability of connection
        prbC = 0.05

        # synapses from FS to RS:
        S_12 = Synapses(G1, G2, on_pre='GsynI_post+=Qi')  # 'v_post -= 1.*mV')
        S_12.connect('i!=j', p=prbC)
        # synapses from FS to FS:
        S_11 = Synapses(G1, G1, on_pre='GsynI_post+=Qi')
        S_11.connect('i!=j', p=prbC)
        # synapses from RS to FS:
        S_21 = Synapses(G2, G1, on_pre='GsynE_post+=Qe')
        S_21.connect('i!=j', p=prbC)
        # synapses from RS to RS:
        S_22 = Synapses(G2, G2, on_pre='GsynE_post+=Qe')
        S_22.connect('i!=j', p=prbC)

        # synapses from external drive to both populations:
        S_ed_in = Synapses(P_ed, G1, on_pre='GsynE_post+=Qe')
        S_ed_in.connect(p=prbC)

        S_ed_ex = Synapses(P_ed, G2, on_pre='GsynE_post+=Qe')
        S_ed_ex.connect(p=prbC)

        ######################################################
        # set recording during simulation
        # number of neuron record of each population:
        Nrecord = 1

        M1G1 = SpikeMonitor(G1)
        M2G1 = StateMonitor(G1, 'v', record=range(Nrecord))
        M3G1 = StateMonitor(G1, 'w', record=range(Nrecord))
        FRG1 = PopulationRateMonitor(G1)

        M1G2 = SpikeMonitor(G2)
        M2G2 = StateMonitor(G2, 'v', record=range(Nrecord))
        M3G2 = StateMonitor(G2, 'w', record=range(Nrecord))
        FRG2 = PopulationRateMonitor(G2)

        #######################################################
        # Run the simulation
        print('simu #' + str(Cnt))
        # print('--##Start simulation##--')
        run(duration)


        # print('--##End simulation##--')

        #######################################################
        # Save recorded data

        # Calculate population friing rate :

        # function for binning:
        def bin_array(array, BIN, time_array):
            N0 = int(BIN / (time_array[1] - time_array[0]))
            N1 = int((time_array[-1] - time_array[0]) / BIN)
            return array[:N0 * N1].reshape((N1, N0)).mean(axis=1)


        BIN = 5
        time_array = np.arange(int(TotTime / DT)) * DT

        LfrG2 = np.array(FRG2.rate / Hz)
        TimBinned, popRateG2 = bin_array(time_array, BIN, time_array), bin_array(LfrG2, BIN, time_array)

        LfrG1 = np.array(FRG1.rate / Hz)
        TimBinned, popRateG1 = bin_array(time_array, BIN, time_array), bin_array(LfrG1, BIN, time_array)

        np.save('Results2/AD_popRateInh_ExIn_' + str(ExIn) + '_bval_' + str(bVal) + 'Nseed_' + str(0) + '.npy',
                popRateG1)
        np.save('Results2/AD_popRateExc_ExIn_' + str(ExIn) + '_bval_' + str(bVal) + 'Nseed_' + str(0) + '.npy',
                popRateG2)
