import matplotlib.pylab as plt
import itertools
import numpy as np
from nest_elephant_tvb.Tvb.modify_tvb.Zerlaut import ZerlautAdaptationSecondOrder as model
from nest_elephant_tvb.Tvb.modify_tvb.fitting_TF.New_parameters import excitatory, inhibitory
from pyqtgraph.Qt import QtGui
import pyqtgraph as pg
import pyqtgraph.opengl as gl


def print_adaptation_effect(parameters,max_frequency=40.0,precision=0.5,frequency=None,rescale=None,
                            name_file ='/home/kusch/Documents/project/co_simulation/co-simulation_mouse/test_nest/test_file/fitting/'):
    app = QtGui.QApplication([])
    window = gl.GLViewWidget()
    window.show()
    window.setWindowTitle('pyqtgraph example: GLSurfacePlot')
    window.setCameraPosition(distance=50)




    for name,value in parameters.items():
        name_file += name+'_'+str(value)+'/'
    if frequency is None:
        frequency = np.arange(0.0,max_frequency,precision)
    frequencies = np.array(list(itertools.product(frequency,frequency)))
    fig1 = plt.figure(); ax1 = fig1.add_subplot(111, projection='3d')
    fig2 = plt.figure(); ax2 = fig2.add_subplot(111)
    fig3 = plt.figure(); ax3 = fig3.add_subplot(111)
    fig4 = plt.figure(); ax4 = fig4.add_subplot(111)
    for w in np.arange(0,100,10):
        #Compute mean of value for the model
        nb_freq = len (frequency)
        if rescale is not None:
            index = np.arange(0,nb_freq,1)
            index = index[np.where(np.logical_not(np.isin(index,rescale)))]
            frequencies =  np.array(list(itertools.product(frequency[index],frequency[index])))
            nb_freq -=len(rescale)
        muV, sV, Tv =model.get_fluct_regime_vars(frequencies[:,1]*1e-3,frequencies[:,0]*1e-3,0.0,0.0,w,parameters['Q_e'],parameters['tau_syn_ex'],parameters['E_ex'],parameters['Q_i'],parameters['tau_syn_in'],parameters['E_in'],
                              parameters['g_L'],parameters['C_m'],parameters['E_L'],parameters['N_tot'],parameters['p_connect'],parameters['g'],0.0,0.0)
        for i in range(nb_freq):
            pts = np.vstack([muV[i*nb_freq:(i+1)*nb_freq],sV[i*nb_freq:(i+1)*nb_freq],Tv[i*nb_freq:(i+1)*nb_freq]]).transpose()
            a = gl.GLLinePlotItem(pos=pts ,color=pg.glColor((w,100*1.3)), width=(w+1)/100., antialias=True)
            window.addItem(a)
        ax1.scatter(muV, sV, Tv*parameters['g_L']/parameters['C_m'], marker='x',s=0.1);ax1.set_xlabel('Vm');ax1.set_ylabel('sV');ax1.set_zlabel('Tv')
        ax2.plot(frequencies[:,0],muV,'x',markersize=0.5)
        ax3.plot(frequencies[:,0],sV,'x',markersize=0.5)
        ax4.plot(frequencies[:,0],Tv*parameters['g_L']/parameters['C_m'],'x',markersize=0.5)
    QtGui.QApplication.instance().exec_()
    plt.show()

print_adaptation_effect(excitatory,40.0,0.2)
print_adaptation_effect(inhibitory,40.0,0.5)