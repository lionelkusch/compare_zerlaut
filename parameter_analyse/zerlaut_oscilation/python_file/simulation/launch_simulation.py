#  Copyright 2023 Aix-Marseille Université
# "Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements; and to You under the Apache License, Version 2.0. "
import os
import numpy as np
from parameter_analyse.zerlaut_oscilation.python_file.analysis.insert_database import init_database
import pathos.multiprocessing as mp
import dill


def run_rate_deterministe(rate_frequency):
    """
    run deterministic simulation
    :param rate_frequency: list of parameters
    :return:
    """
    # parameters of the function
    rate = rate_frequency['rate']
    frequency = rate_frequency['frequency']
    path_simulation = rate_frequency['path']
    duration = rate_frequency['duration']
    database = rate_frequency['database']
    table_name = rate_frequency['table_name']

    # import function
    import parameter_analyse.zerlaut_oscilation.python_file.run.tools_simulation as tools
    from parameter_analyse.zerlaut_oscilation.python_file.analysis.analysis import analysis
    from parameter_analyse.zerlaut_oscilation.python_file.analysis.insert_database import\
        insert_database, check_already_analyse_database
    from parameter_analyse.zerlaut_oscilation.python_file.parameters.parameter_default import Parameter
    parameters = Parameter()
    parameters.parameter_integrator['stochastic'] = False
    parameters.parameter_model['initial_condition']["external_input_excitatory_to_excitatory"] = [rate * 1e-3,
                                                                                                  rate * 1e-3]
    parameters.parameter_model['initial_condition']["external_input_excitatory_to_inhibitory"] = [rate * 1e-3,
                                                                                                  rate * 1e-3]
    parameters.parameter_stimulus['frequency'] = frequency * 1e-3
    parameters.parameter_stimulus['amp'] = (np.concatenate((np.arange(0.1, 1.5, 0.1),
                                                            np.arange(1.5, 7., 0.5)
                                                            )) * 1e-3).tolist()
    parameters.parameter_connection_between_region['number_of_regions'] = len(parameters.parameter_stimulus['amp'])
    parameters.parameter_monitor['Raw'] = True
    parameters.parameter_simulation['path_result'] = path_simulation + "/rate_" + str(rate) \
                                                     + "/frequency_" + str(frequency) + "/"
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
                             duration,
                             parameters.parameter_simulation,
                             parameters.parameter_monitor)
    if not check_already_analyse_database(database, table_name,
                                          parameters.parameter_simulation['path_result'], 'excitatory'):
        results = analysis(parameters.parameter_simulation['path_result'], begin=2500.0, end=duration, init_remove=0)
        insert_database(database, table_name, results)


def run_rate_stochastic(rate_frequency):
    """
    run stochastic
    :param rate_frequency: parameters of the frequency
    :return:
    """
    # parameters
    rate = rate_frequency['rate']
    frequency = rate_frequency['frequency']
    noise = rate_frequency['noise']
    path_simulation = rate_frequency['path']
    duration = rate_frequency['duration']
    database = rate_frequency['database']
    table_name = rate_frequency['table_name']

    # import
    import parameter_analyse.zerlaut_oscilation.python_file.run.tools_simulation as tools
    from parameter_analyse.zerlaut_oscilation.python_file.analysis.analysis import analysis
    from parameter_analyse.zerlaut_oscilation.python_file.analysis.insert_database import \
        insert_database, check_already_analyse_database
    from parameter_analyse.zerlaut_oscilation.python_file.parameters.parameter_default import Parameter

    # parameters
    parameters = Parameter()
    parameters.parameter_integrator['stochastic'] = True
    parameters.parameter_integrator['noise_parameter']['nsig'][0] = noise
    parameters.parameter_integrator['noise_parameter']['nsig'][1] = noise
    parameters.parameter_model['initial_condition']["external_input_excitatory_to_excitatory"] = [rate * 1e-3,
                                                                                                  rate * 1e-3]
    parameters.parameter_model['initial_condition']["external_input_excitatory_to_inhibitory"] = [rate * 1e-3,
                                                                                                  rate * 1e-3]
    parameters.parameter_stimulus['frequency'] = frequency * 1e-3
    parameters.parameter_stimulus['amp'] = (np.concatenate((np.arange(0.1, 1.5, 0.1),
                                                            np.arange(1.5, 7., 0.5)
                                                            )) * 1e-3).tolist()
    parameters.parameter_connection_between_region['number_of_regions'] = len(parameters.parameter_stimulus['amp'])
    parameters.parameter_monitor['Raw'] = True
    parameters.parameter_simulation['path_result'] = path_simulation + "/rate_" + str(rate) \
                                                     + "/frequency_" + str(frequency) + "/"
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
                             duration,
                             parameters.parameter_simulation,
                             parameters.parameter_monitor)
    if not check_already_analyse_database(database, table_name,
                                          parameters.parameter_simulation['path_result'], 'excitatory'):
        results = analysis(parameters.parameter_simulation['path_result'], begin=2500.0, end=duration, init_remove=0)
        insert_database(database, table_name, results)


if __name__ == "__main__":
    # # test 1
    # run_rate_deterministe({
    #     'rate': 7.0, 'frequency': 30.0, 'duration': 20001.0,
    #     'path': os.path.dirname(os.path.realpath(__file__)) + '/../../simulation/deterministe/',
    #     'database': os.path.dirname(os.path.realpath(__file__)) + '/../../simulation/deterministe/database.db',
    #     'table_name': "exploration"
    #     })


    p = mp.ProcessingPool(ncpus=8)
    path_simulation = os.path.dirname(os.path.realpath(__file__)) + '/../../simulation/deterministe/'
    database = path_simulation + "/database.db"
    table_name = "exploration"
    duration = 20001.0
    init_database(database, table_name)
    for rate in [7.0, 0.0, 2.5]:
        list_parameters = []
        if not os.path.exists(path_simulation + "/rate_" + str(rate)):
            os.mkdir(path_simulation + "/rate_" + str(rate))
        for frequency in np.concatenate(([1], np.arange(5., 51., 5.))):
            list_parameters.append({'rate': rate, 'frequency': frequency, 'path': path_simulation,
                                     'duration': duration, 'database': database, 'table_name': table_name })
        p.map(dill.copy(run_rate_deterministe), list_parameters)

    for rate in [7.0, 0.0, 2.5]:
        list_parameters = []
        for noise in [1e-9, 1e-8]:
            path_simulation = os.path.dirname(os.path.realpath(__file__)) +\
                              '/../../simulation/stochastic_' + str(noise) + '/'
            database = path_simulation + "/database.db"
            if not os.path.exists(path_simulation + "/rate_" + str(rate)):
                os.mkdir(path_simulation + "/rate_" + str(rate))
            init_database(database, table_name)
            for frequency in np.concatenate(([1], np.arange(5., 51., 5.))):
                list_parameters.append(
                    {'noise': noise, 'rate': rate, 'frequency': frequency, 'path': path_simulation,
                     'duration': duration, 'database': database, 'table_name': table_name})
        p.map(dill.copy(run_rate_stochastic), list_parameters)

