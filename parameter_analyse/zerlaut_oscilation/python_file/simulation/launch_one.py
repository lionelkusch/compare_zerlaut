import numpy as np
import os
import parameter_analyse.zerlaut_oscilation.python_file.run.tools_simulation as tools
from parameter_analyse.zerlaut_oscilation.python_file.parameters.parameter_default import Parameter


def run_rate_deterministe(rate_frequency, end=200.0):
    """
    run one example
    :param rate_frequency: list of parameters
    :param end: duration of simulation
    :return:
    """
    # gte parameters
    rate = rate_frequency['rate']
    frequency = rate_frequency['frequency']
    path_simulation = rate_frequency['path']

    # define parameters
    parameters = Parameter()
    parameters.parameter_simulation['path_result'] = path_simulation
    parameters.parameter_integrator['stochastic'] = False
    parameters.parameter_model['initial_condition']["external_input_excitatory_to_excitatory"] = [rate* 1e-3,
                                                                                                  rate * 1e-3]
    parameters.parameter_model['initial_condition']["external_input_excitatory_to_inhibitory"] = [rate * 1e-3,
                                                                                                  rate * 1e-3]
    parameters.parameter_stimulus['frequency'] = frequency * 1e-3
    parameters.parameter_stimulus['amp'] = (np.arange(0.0, rate+0.1, 0.1) * 1e-6).tolist()
    parameters.parameter_connection_between_region['number_of_regions'] = len(parameters.parameter_stimulus['amp'])
    parameters.parameter_simulation['path_result'] = path_simulation + "/rate_" + str(rate) \
                                                     + "/frequency_" + str(frequency)
    counter = 0
    while os.path.exists(parameters.parameter_simulation['path_result']):
        counter += 1
        parameters.parameter_simulation['path_result'] = path_simulation + '_' + str(counter)
    parameters.parameter_simulation['path_result'] += '/'
    print(parameters.parameter_simulation['path_result'])
    simulator = tools.init(parameters.parameter_simulation,
                           parameters.parameter_model,
                           parameters.parameter_connection_between_region,
                           parameters.parameter_coupling,
                           parameters.parameter_integrator,
                           parameters.parameter_monitor,
                           parameter_stimulation=parameters.parameter_stimulus)
    tools.run_simulation(simulator,
                         end,
                         parameters.parameter_simulation,
                         parameters.parameter_monitor)
    return parameters


if __name__ == "__main__":
    from parameter_analyse.zerlaut_oscilation.python_file.print.print_one import print_result

    path_simulation = '/home/kusch/Documents/project/Zerlaut/compare_zerlaut/parameter_analyse/zerlaut_oscilation/simulation/deterministe/test/'
    list_parameters = []
    end = 5001.0
    for rate in [7.0]:
        if not os.path.exists(path_simulation + "/rate_" + str(rate)):
            os.mkdir(path_simulation + "/rate_" + str(rate))
        for frequency in [1]:
            parameters = run_rate_deterministe({'rate': rate, 'frequency': frequency, 'path': path_simulation}, end=end)
            print_result(parameters.parameter_simulation['path_result'], begin=0.0, end=end)
