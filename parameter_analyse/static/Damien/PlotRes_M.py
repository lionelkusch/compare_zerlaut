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
Lb=[i*2 for i in range(60)]
Lei=[i for i in range(100)]

hmRes=[]
FRmu=[]
Div=[]
for bVal in Lb:
	hmRes.append([])
	FRmu.append([])
	Div.append([])
	for ExIn in Lei:

		FRinh = np.load('Results2/AD_popRateInh_ExIn_'+str(ExIn)+'_bval_'+str(bVal)+'Nseed_'+str(0)+'.npy')
		FRexc = np.load('Results2/AD_popRateExc_ExIn_'+str(ExIn)+'_bval_'+str(bVal)+'Nseed_'+str(0)+'.npy')
		hmRes[Cnt].append(np.mean(FRexc[100::]))
		#FRmu[Cnt].append(np.mean(FRinh))
		#Div[Cnt].append(np.std(FRinh)/np.mean(FRinh))
	Cnt+=1

#plt.figure()
#plt.imshow(Div, origin='lower')

plt.figure()
plt.imshow(hmRes, origin='lower', extent=[0,100,0,120])
plt.xlabel('External input FR', fontsize=16)
plt.ylabel('b value', fontsize=20)
clb=plt.colorbar()#orientation="horizontal")
clb.ax.tick_params(labelsize=8) 
clb.ax.set_ylabel('Mean FR (Hz)',fontsize=18)#, rotation=270)
clb.ax.tick_params(labelsize=15)
#plt.figure()
#plt.imshow(FRmu, origin='lower')
#plt.colorbar()
plt.figure()
FRinh = np.load('Results2/AD_popRateInh_ExIn_'+str(60)+'_bval_'+str(100)+'Nseed_'+str(0)+'.npy')
FRexc = np.load('Results2/AD_popRateExc_ExIn_'+str(60)+'_bval_'+str(100)+'Nseed_'+str(0)+'.npy')
plt.plot(FRinh[100::], 'r')
plt.plot(FRexc[100::], 'g')
plt.show()
