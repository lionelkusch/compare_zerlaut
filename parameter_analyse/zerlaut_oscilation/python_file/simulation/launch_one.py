#  Copyright 2023 Aix-Marseille Université
# "Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements; and to You under the Apache License, Version 2.0. "
import os
import numpy as np
import parameter_analyse.zerlaut_oscilation.python_file.run.tools_simulation as tools
from parameter_analyse.zerlaut_oscilation.python_file.parameters.parameter_default import Parameter


def run_rate_deterministe(rate_frequency, b=0.0, end=200.0, init_E=[0.0, 0.0], init_I=[0.0, 0.0], T=5.0):
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
    parameters.parameter_model['T'] = T
    parameters.parameter_model['b_e'] = b
    parameters.parameter_model['initial_condition']['E'] = init_E
    parameters.parameter_model['initial_condition']['I'] = init_I
    # parameters.parameter_model['initial_condition']['W_e'] = [0., 0.]
    parameters.parameter_model['initial_condition']["external_input_excitatory_to_excitatory"] = [rate * 1e-3,
                                                                                                  rate * 1e-3]
    # parameters.parameter_model['initial_condition']["external_input_excitatory_to_inhibitory"] = [rate * 1e-3,
    #                                                                                               rate * 1e-3]
    parameters.parameter_stimulus ['variables'] = [8]
    # parameters.parameter_model['initial_condition']["W_e"] = [0.0, 0.0]
    parameters.parameter_stimulus['frequency'] = frequency * 1e-3
    # parameters.parameter_stimulus['amp'] = (np.arange(0.0, rate+0.1, 0.1) * 1e-3).tolist()
    # parameters.parameter_stimulus['amp'] = (np.arange(0.0, 5.0, 0.1) * 1e-3).tolist()
    # parameters.parameter_stimulus['amp'] = [0.0]
    parameters.parameter_stimulus['amp'] = [3.0 * 1e-3]
    parameters.parameter_connection_between_region['number_of_regions'] = len(parameters.parameter_stimulus['amp'])
    parameters.parameter_simulation['path_result'] = path_simulation + "/b_"+str(b) + "/rate_" + str(rate) \
                                                     + "/frequency_" + str(frequency)
    counter = 0
    while os.path.exists(parameters.parameter_simulation['path_result']):
        counter += 1
        parameters.parameter_simulation['path_result'] = "/b_"+str(b) \
                                                         + "/rate_" + str(rate) \
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

    path_simulation = os.path.dirname(os.path.realpath(__file__)) + '/../../simulation/deterministe/short_b/'
    rate = 25.0
    end = 4001.0
    for rate, init_E, init_I in [(25.0, [1.3501687710963872e-3, 1.3501687710963872e-3], [95.34767383691845e-3, 95.34767383691845e-3]),
                                 (50.0, [3.4504313039129886e-3, 3.4504313039129886e-3], [152.87643821910953e-3, 152.87643821910953e-3]),
                                 (75.0, [3.3504188023502937e-3, 3.3504188023502937e-3], [200.10005002501248e-3, 200.10005002501248e-3]),]:

        for b in [0.0, 30.0, 60.0]:
            if not os.path.exists(path_simulation + "/b_"+str(b)):
                os.mkdir(path_simulation + "/b_"+str(b))
            if not os.path.exists(path_simulation + "/b_"+str(b) + "/rate_" + str(rate)):
                os.mkdir(path_simulation + "/b_"+str(b) + "/rate_" + str(rate))
            for frequency in [0.0]:
                print(rate, frequency)
                if not os.path.exists(path_simulation + "/b_"+str(b) + "/rate_" + str(rate) + "/frequency_" + str(frequency)):
                    parameters = run_rate_deterministe({'rate': rate, 'frequency': frequency, 'path': path_simulation}, b=b,
                                                       end=end, init_E=init_E, init_I=init_I)
            #     plot_result(path_simulation + "/b_"+str(b) + "/rate_" + str(rate) + "/frequency_" + str(frequency),
            #                  begin=0.0, end=4000.0, region=0)
            # plt.show()
    path_simulation = os.path.dirname(os.path.realpath(__file__)) + '/../../simulation/deterministe/review/'
    rate = 25.0
    end = 4001.0
    for b in [0.0, 30.0, 60.0]:
         for rate, init_E, init_I in [(25.0, [1.3501687710963872e-3, 1.3501687710963872e-3], [95.34767383691845e-3, 95.34767383691845e-3]),
                                 (50.0, [3.4504313039129886e-3, 3.4504313039129886e-3], [152.87643821910953e-3, 152.87643821910953e-3]),
                                 (75.0, [3.3504188023502937e-3, 3.3504188023502937e-3], [200.10005002501248e-3, 200.10005002501248e-3]),]:
                for T in [1.0, 2.0]:
                    if not os.path.exists(path_simulation + '/T_'+str(T)):
                        os.mkdir(path_simulation + '/T_'+str(T))
                    if not os.path.exists(path_simulation + '/T_'+str(T) + "/b_"+str(b)):
                        os.mkdir(path_simulation + '/T_'+str(T)+ "/b_"+str(b))
                    if not os.path.exists(path_simulation + '/T_'+str(T) + "/b_"+str(b) + "/rate_" + str(rate)):
                        os.mkdir(path_simulation + '/T_'+str(T)+ "/b_"+str(b) + "/rate_" + str(rate))
                    for frequency in [0.0]:
                        print(rate, frequency)
                        if not os.path.exists(path_simulation + '/T_'+str(T)+ "/b_"+str(b) + "/rate_" + str(rate) + "/frequency_" + str(frequency)):
                            parameters = run_rate_deterministe({'rate': rate, 'frequency': frequency, 'path': path_simulation + '/T_'+str(T)}, b=b,
                                                               end=end, init_E=init_E, init_I=init_I, T=T)
                        # plot_result(path_simulation +'/T_'+str(T)+ "/b_"+str(b) + "/rate_" + str(rate) + "/frequency_" + str(frequency),
                        #              begin=0.0, end=4000.0, region=0)
    end = 2001.0
    b= 0.0
    for rate, init_E, init_I in [(10.0, [0.000125, 0.000125], [0.05, 0.05]),
                                  # (60.0, [0.00362, 0.00362], [0.17959, 0.17959]),
                                  (80.0, [0.0041, 0.0041], [0.2001, 0.20001]),
                                  ]:
            for T in [1.0, 2.0]:
                if not os.path.exists(path_simulation + '/T_'+str(T)):
                    os.mkdir(path_simulation + '/T_'+str(T))
                if not os.path.exists(path_simulation + '/T_'+str(T) + "/b_"+str(b)):
                    os.mkdir(path_simulation + '/T_'+str(T)+ "/b_"+str(b))
                if not os.path.exists(path_simulation+ '/T_'+str(T)+ "/b_"+str(b) + "/rate_" + str(rate)):
                    os.mkdir(path_simulation+ '/T_'+str(T)+ "/b_"+str(b) + "/rate_" + str(rate))
                for frequency in [0.0]:
                    print(rate, frequency)
                    if not os.path.exists(path_simulation+ '/T_'+str(T)+ "/b_"+str(b) + "/rate_" + str(rate) + "/frequency_" + str(frequency)):
                        parameters = run_rate_deterministe({'rate': rate, 'frequency': frequency, 'path': path_simulation+ '/T_'+str(T)},
                                                           end=end, init_E=init_E, init_I=init_I, T=T)
                    # plot_result(path_simulation + "/rate_" + str(rate) + "/frequency_" + str(frequency),
                    #              begin=0.0, end=2000.0, region=0)
    plt.show()


    # Impact of T
    end = 4001.0
    b = 0.0
    for rate, init_E, init_I in [(0.0, [0.0, 0.0], [0.0, 0.0]),
                                 (7.0, [0.0, 0.0], [0.0, 0.0])]:
            for T in [0.2, 0.5, 1, 5, 50, 150]:
                path_simulation = os.path.dirname(os.path.realpath(__file__)) + '/../../simulation/deterministe/T_'+str(T)+'/'
                for frequency in [5.0, 20.0]:
                    print(rate, T, frequency)
                    if not os.path.exists(path_simulation + "/b_"+str(b)+"/rate_" + str(rate) + "/frequency_" + str(frequency)):
                        if not os.path.exists(path_simulation + '/b_'+str(b)):
                            os.mkdir(path_simulation + '/b_'+str(b))
                        if not os.path.exists(path_simulation + '/b_'+str(b)+"/rate_" + str(rate)):
                            os.mkdir(path_simulation + '/b_'+str(b)+"/rate_" + str(rate))
                        parameters = run_rate_deterministe({'rate': rate, 'frequency': frequency, 'path': path_simulation},
                                                           end=end, init_E=init_E, init_I=init_I, T=T)
                    plot_result(path_simulation + '/b_'+str(b) + "/rate_" + str(rate) + "/frequency_" + str(frequency),
                                 begin=0.0, end=2000.0, region=0)
                    # plot_result(path_simulation + "/rate_" + str(rate) + "/frequency_" + str(frequency),
                    #             begin=0.0, end=2000.0, region=17)
                    # plt.show()


    from parameter_analyse.static.python_file.analysis.analysis_global import get_gids, load_spike, slidding_window

    end = 4001.0
    b = 0.0
    for rate, init_E, init_I in [(0.0, [0.0, 0.0], [0.0, 0.0]),
                                 (7.0, [0.0, 0.0], [0.0, 0.0])]:
        for T in [0.2, 0.5, 1, 5, 50, 150]:
            path_simulation = os.path.dirname(os.path.realpath(__file__)) + '/../../simulation/deterministe/T_' + str(
                T) + '_init/'
            if not os.path.exists(path_simulation):
                os.mkdir(path_simulation)
            for frequency in [5.0, 20.0]:
                path_simulation_network = os.path.dirname(os.path.realpath(__file__)) + '/../../../spike_oscilation/simulation/simulation/rate_' +\
                                          str(rate) + '/_frequency_' + str(frequency) + '_amplitude_3.0/'
                gids = get_gids(path_simulation_network, 0)  # 0 excitatory
                data_all = load_spike(gids, path_simulation_network, 1000., 2000., 0)
                init_E = len(data_all) / 1000.0 * 1e-5
                gids = get_gids(path_simulation_network, 1)  # 1 inhibitory
                data_all = load_spike(gids, path_simulation_network, 1000., 2000., 1)
                init_I = len(data_all) / 1000.0 * 1e-5
                print(rate, T, frequency, init_E, init_I)
                if not os.path.exists(
                        path_simulation + "/b_" + str(b) + "/rate_" + str(rate) + "/frequency_" + str(frequency)):
                    if not os.path.exists(path_simulation + '/b_' + str(b)):
                        os.mkdir(path_simulation + '/b_' + str(b))
                    if not os.path.exists(path_simulation + '/b_' + str(b) + "/rate_" + str(rate)):
                        os.mkdir(path_simulation + '/b_' + str(b) + "/rate_" + str(rate))
                    parameters = run_rate_deterministe({'rate': rate, 'frequency': frequency, 'path': path_simulation},
                                                       end=end, init_E=[init_E,init_E], init_I=[init_I, init_I], T=T)
                plot_result(path_simulation + '/b_' + str(b) + "/rate_" + str(rate) + "/frequency_" + str(frequency),
                            begin=0.0, end=2000.0, region=0)
                # plot_result(path_simulation + "/rate_" + str(rate) + "/frequency_" + str(frequency),
                #             begin=0.0, end=2000.0, region=17)
                # plt.show()


    end = 4001.0
    b = 0.0
    for rate, init_E, init_I in [(0.0, [0.0, 0.0], [0.0, 0.0]),
                                 (7.0, [0.0, 0.0], [0.0, 0.0])]:
        for T in [0.2, 0.5, 1, 5, 50, 150]:
            path_simulation = os.path.dirname(os.path.realpath(__file__)) + '/../../simulation/deterministe/T_' + str(
                T) + '_ex/'
            if not os.path.exists(path_simulation):
                os.mkdir(path_simulation)
            for frequency in [5.0, 20.0]:
                print(rate, T, frequency, init_E, init_I)
                if not os.path.exists(
                        path_simulation + "/b_" + str(b) + "/rate_" + str(rate) + "/frequency_" + str(frequency)):
                    if not os.path.exists(path_simulation + '/b_' + str(b)):
                        os.mkdir(path_simulation + '/b_' + str(b))
                    if not os.path.exists(path_simulation + '/b_' + str(b) + "/rate_" + str(rate)):
                        os.mkdir(path_simulation + '/b_' + str(b) + "/rate_" + str(rate))
                    parameters = run_rate_deterministe({'rate': rate, 'frequency': frequency, 'path': path_simulation},
                                                       end=end, init_E=init_E, init_I=init_I, T=T)
                # plot_result(path_simulation + '/b_' + str(b) + "/rate_" + str(rate) + "/frequency_" + str(frequency),
                #             begin=0.0, end=4000.0, region=0)
                # plot_result(os.path.dirname(os.path.realpath(__file__)) + '/../../simulation/deterministe/T_' + str(
                # T) + '/' + '/b_' + str(b) + "/rate_" + str(rate) + "/frequency_" + str(frequency),
                #             begin=0.0, end=4000.0, region=0)
                # plt.show()

