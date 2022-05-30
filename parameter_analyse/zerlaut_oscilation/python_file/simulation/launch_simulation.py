import numpy as np
import os
import parameter_analyse.zerlaut_oscilation.python_file.run.tools_simulation as tools
from parameter_analyse.zerlaut_oscilation.python_file.parameters.parameter_default import Parameter
import matplotlib.pyplot as plt

path_simulation = '/home/kusch/Documents/project/Zerlaut/compare_zerlaut/parameter_analyse/zerlaut_oscilation/simulation/'

# for noise in np.arange(1e-9, 1e-8, 1e-9):
# for noise in np.arange(1e-8, 1e-7, 1e-8):
# for noise in np.arange(0.0, 1e-5, 5e-7):
#     for frequency in np.concatenate(([1], np.arange(5., 51., 5.))):
for noise in [1e-8]:
    for frequency in [1]:
        parameters = Parameter()
        parameters.parameter_integrator['noise_parameter']['dt'] = 0.001
        parameters.parameter_integrator['dt'] = 0.001

        parameters.parameter_stimulus['frequency'] = frequency*1e-3
        parameters.parameter_integrator['noise_parameter']['nsig'][0] = noise
        parameters.parameter_integrator['noise_parameter']['nsig'][1] = noise
        parameters.parameter_simulation['path_result'] = path_simulation + "aprecise_frequency_"+str(frequency)+"_noise_"+str(noise)+"/"

        print(parameters.parameter_simulation['path_result'])
        if not os.path.exists(parameters.parameter_simulation['path_result']):


            simulator = tools.init(parameters.parameter_simulation,
                                   parameters.parameter_model,
                                   parameters.parameter_connection_between_region,
                                   parameters.parameter_coupling,
                                   parameters.parameter_integrator,
                                   parameters.parameter_monitor,
                                   parameter_stimulation=parameters.parameter_stimulus)

            tools.run_simulation(simulator,
                                 20000.0,
                                 parameters.parameter_simulation,
                                 parameters.parameter_monitor)

# result = tools.get_result(parameters.parameter_simulation['path_result'], 0.0, 20000.0)
#
# times = result[0][0]
# rateE = result[0][1][:, 0, :]
# stdE = result[0][1][:, 2, :]
# rateI = result[0][1][:, 1, :]
# stdI = result[0][1][:, 4, :]
# corrEI = result[0][1][:, 3, :]
# adaptationE = result[0][1][:, 5, :]
#
# #Plot the excitatory and inhibitory signals, excitatory neuron adaptation, and noise input
# # to the stimulated node
# for i in range(50):
#     plt.figure(figsize=(20,4))
#     plt.rcParams.update({'font.size': 14})
#     ax0 = plt.subplot(211)
#     ax0.plot(result[0][0]*1e-3,result[0][1][:,0,i]*1e3, 'c')
#     ax0.plot(result[0][0]*1e-3,result[0][1][:,1,i]*1e3, 'r')
#     ax0.set_xlabel('time [s]')
#     ax0.set_ylabel('firing rate [Hz]')
#     ax1 = plt.subplot(212)
#     ax1.plot(result[0][0]*1e-3,result[0][1][:,5,i], 'k')
#     ax1.set_xlabel('time [s]')
#     ax1.set_ylabel('adaptation [nA]')
# plt.show()