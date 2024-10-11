#  Copyright 2023 Aix-Marseille Université
# "Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements; and to You under the Apache License, Version 2.0. "
import os
import numpy as np
import parameter_analyse.zerlaut_oscilation.python_file.run.tools_simulation as tools
from parameter_analyse.zerlaut_oscilation.python_file.parameters.parameter_default import Parameter


def run_rate_deterministe(rate_frequency, end=200.0, init_E=[0.0, 0.0], init_I=[0.0, 0.0]):
    """
    run one example
    :param rate_frequency: list of parameters
    :param end: duration of simulation
    :param init_E: initial condition of Excitatory population
    :param init_I: initial condition of Inhibitory population
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
    parameters.parameter_model['initial_condition']['E'] = init_E
    parameters.parameter_model['initial_condition']['I'] = init_I
    parameters.parameter_model['initial_condition']["external_input_excitatory_to_excitatory"] = [rate * 1e-3,
                                                                                                  rate * 1e-3]
    parameters.parameter_model['initial_condition']["external_input_excitatory_to_inhibitory"] = [rate * 1e-3,
                                                                                                  rate * 1e-3]
    parameters.parameter_model['initial_condition']["W_e"] = [0.0, 0.0]
    parameters.parameter_stimulus['frequency'] = frequency * 1e-3
    # parameters.parameter_stimulus['amp'] = (np.arange(0.0, rate+0.1, 0.1) * 1e-3).tolist()
    # parameters.parameter_stimulus['amp'] = (np.arange(0.0, 5.0, 0.1) * 1e-3).tolist()
    parameters.parameter_stimulus['amp'] = [0.0]
    parameters.parameter_connection_between_region['number_of_regions'] = len(parameters.parameter_stimulus['amp'])
    parameters.parameter_simulation['path_result'] = path_simulation + "/rate_" + str(rate) \
                                                     + "/frequency_" + str(frequency)
    counter = 0
    while os.path.exists(parameters.parameter_simulation['path_result']):
        counter += 1
        parameters.parameter_simulation['path_result'] = "/rate_" + str(rate) \
                                                         + "/frequency_" + str(frequency) + '_' + str(counter)
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
    from parameter_analyse.zerlaut_oscilation.python_file.print.print_one import plot_result
    import matplotlib.pyplot as plt

    path_simulation = os.path.dirname(os.path.realpath(__file__)) + '/../../simulation/deterministe/instability/'
    list_parameters = []
    end = 2001.0
    range_rate = [0.0, 0.2, 0.3, 0.4, 0.6, 1.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0]
    range_frequency = [0.0]
    for rate in range_rate:
        if not os.path.exists(path_simulation + "/rate_" + str(rate)):
            os.mkdir(path_simulation + "/rate_" + str(rate))
        for frequency in range_frequency:
            if not os.path.exists(path_simulation + "/rate_" + str(rate) + "/frequency_" + str(frequency)):
                parameters = run_rate_deterministe({'rate': rate, 'frequency': frequency, 'path': path_simulation},
                                                   end=end)

    for rate in range_rate:
        for frequency in range_frequency:
            print(rate, frequency)
            plot_result(path_simulation + "/rate_" + str(rate) + "/frequency_" + str(frequency), begin=0.0, end=end)
    plt.show()

    path_simulation = os.path.dirname(os.path.realpath(__file__)) + '/../../simulation/deterministe/short/'
    for rate, init_E, init_I in [(10.0, [0.000125, 0.000125], [0.05, 0.05]),
                                 # (60.0, [0.00362, 0.00362], [0.17959, 0.17959]),
                                 (80.0, [0.0041, 0.0041], [0.2001, 0.20001]),
                                 ]:
        if not os.path.exists(path_simulation + "/rate_" + str(rate)):
            os.mkdir(path_simulation + "/rate_" + str(rate))
        for frequency in [0.0]:
            print(rate, frequency)
            if not os.path.exists(path_simulation + "/rate_" + str(rate) + "/frequency_" + str(frequency)):
                parameters = run_rate_deterministe({'rate': rate, 'frequency': frequency, 'path': path_simulation},
                                                   end=end, init_E=init_E, init_I=init_I)
            plot_result(path_simulation + "/rate_" + str(rate) + "/frequency_" + str(frequency),
                         begin=0.0, end=2000.0, region=0)
        plt.show()
