# import libraries
import matplotlib.pyplot as plt
import numpy as np
from brian2 import *
import matplotlib.pyplot as plt

plt.rc('xtick',labelsize=15)
plt.rc('ytick',labelsize=15)


Cnt=0
#Lb=[i*2 for i in range(60)]
#Lei=[i*0.1 for i in range(20)]
Lb= [i*2 for i in range(60)]
# Lb = [0, 30, 60]
Lei=[i for i in range(100)]

hmResExc=[]
FRmuExc=[]
hmResInh=[]
FRmuInh=[]

Div=[]
for bVal in Lb:
	hmResExc.append([])
	FRmuExc.append([])
	hmResInh.append([])
	FRmuInh.append([])
	Div.append([])
	for ExIn in Lei:

		FRinh = np.load('Results2/AD_popRateInh_ExIn_'+str(ExIn)+'_bval_'+str(bVal)+'Nseed_'+str(0)+'.npy')
		FRexc = np.load('Results2/AD_popRateExc_ExIn_'+str(ExIn)+'_bval_'+str(bVal)+'Nseed_'+str(0)+'.npy')
		print('Results2/AD_popRateInh_ExIn_'+str(ExIn)+'_bval_'+str(bVal)+'Nseed_'+str(0)+'.npy')
		hmResExc[Cnt].append(np.std(FRexc[500::]))
		FRmuExc[Cnt].append(np.mean(FRexc[500::]))
		hmResInh[Cnt].append(np.std(FRinh[500::]))
		FRmuInh[Cnt].append(np.mean(FRinh[500::]))
		#Div[Cnt].append(np.std(FRinh)/np.mean(FRinh))
	Cnt+=1

	plt.figure()
	#plt.imshow(Div, origin='lower')
	#plt.xlabel('External input FR', fontsize=16)
	#plt.ylabel('Firing Rate (Hz)', fontsize=20)
	plt.subplot(2, 2, 1)
	plt.errorbar(Lei, FRmuExc[0], yerr=hmResExc[0], color='g')
	plt.errorbar(Lei, FRmuInh[0], yerr=hmResInh[0], color='r')
	plt.xlabel('External input FR', fontsize=20)
	plt.ylabel('Firing Rate (Hz)', fontsize=20)
	plt.title('AdEx Network: firing rate in function of the input')
	plt.subplot(2, 2, 2)
	plt.plot(FRmuExc[0], FRmuInh[0], '--o')
	plt.xlabel('Excitatory Firing Rate (Hz)', fontsize=20)
	plt.ylabel('Inhibitory Firing Rate (Hz)', fontsize=20)
	plt.title('AdEx Network: firing rate Inh vs Exc')
	plt.subplot(2, 2, 3)
	plt.errorbar(Lei,FRmuExc[0], yerr=hmResExc[0], color='g', ls='dotted')
	plt.xlabel('External input FR', fontsize=20)
	plt.ylabel('Firing Rate (Hz)', fontsize=20)
	plt.title('AdEx Network: Excitatory firing rate')
	plt.subplot(2, 2, 4)
	plt.errorbar(Lei, FRmuInh[0], yerr=hmResInh[0], color='r', ls='dotted')
	plt.xlabel('External input FR', fontsize=20)
	plt.ylabel('Firing Rate (Hz)', fontsize=20)
	plt.title('AdEx Network: Excitatory firing rate')
	#plt.imshow(hmRes,origin='lower', extent=[0,100,0,120])
	#plt.xlabel('External input FR', fontsize=16)
	#plt.ylabel('b value', fontsize=20)
	#clb=plt.colorbar()#orientation="horizontal")
	#clb.ax.tick_params(labelsize=8)
	#clb.ax.set_ylabel('Standard deviation (Hz)',fontsize=18)#, rotation=270)
	#clb.ax.tick_params(labelsize=15)
	#plt.figure()
	#plt.imshow(FRmu, origin='lower')
	#plt.colorbar()
	#plt.figure()
	#FRinh = np.load('Results2/AD_popRateInh_ExIn_'+str(60)+'_bval_'+str(56)+'Nseed_'+str(0)+'.npy')
	#FRexc = np.load('Results2/AD_popRateExc_ExIn_'+str(60)+'_bval_'+str(56)+'Nseed_'+str(0)+'.npy')
	#plt.plot(FRinh[500::], 'r')
	#plt.plot(FRexc[500::], 'g')
plt.show()
