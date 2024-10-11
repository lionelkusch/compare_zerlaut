#  Copyright 2023 Aix-Marseille Université
# "Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements; and to You under the Apache License, Version 2.0. "
import tvb.simulator.lab as lab
import numpy.random as rgn
import numpy as np
import json
import os
from scipy.optimize import fsolve
import parameter_analyse.zerlaut_oscilation.python_file.run.zerlaut as zerlaut


def init(parameter_simulation, parameter_model, parameter_connection_between_region, parameter_coupling,
         parameter_integrator, parameter_monitor, initial_condition=None, parameter_stimulation=None,
         my_seed=10):
    '''
    Initialise the simulator with parameter

    :param parameter_simulation: parameters for the simulation
    :param parameter_model: parameters for the model
    :param parameter_connection_between_region: parameters for the connection between nodes
    :param parameter_coupling: parameters for the coupling of equations
    :param parameter_integrator: parameters for the intergator of the equation
    :param parameter_monitor: parameters for the monitors
    :param initial_condition: the possibility to add an initial condition
    :param parameter_stimulation: parameter of the stimulation
    :param my_seed: seed of the random generator
    :return: the simulator initialize
    '''
    ## initialise the random generator
    parameter_simulation['seed'] = my_seed
    rgn.seed(parameter_simulation['seed'])


    ## Model
    if parameter_model['order'] == 1:
        model = zerlaut.ZerlautAdaptationFirstOrder(variables_of_interest='E I W_e W_i ou_drift '\
                      'external_input_excitatory_to_excitatory external_input_excitatory_to_inhibitory ' \
                      'external_input_inhibitory_to_inhibitory external_input_inhibitory_to_inhibitory'.split())
    elif parameter_model['order'] == 2:
        model = zerlaut.ZerlautAdaptationSecondOrder(variables_of_interest='E I C_ee C_ei C_ii W_e W_i ou_drift '\
                      'external_input_excitatory_to_excitatory external_input_excitatory_to_inhibitory ' \
                      'external_input_inhibitory_to_inhibitory external_input_inhibitory_to_inhibitory'.split())
    else:
        raise Exception('Bad order for the model')

    model.g_L = np.array(parameter_model['g_L'])
    model.E_L_e = np.array(parameter_model['E_L_e'])
    model.E_L_i = np.array(parameter_model['E_L_i'])
    model.C_m = np.array(parameter_model['C_m'])
    model.b_e = np.array(parameter_model['b_e'])
    model.a_e = np.array(parameter_model['a_e'])
    model.b_i = np.array(parameter_model['b_i'])
    model.a_i = np.array(parameter_model['a_i'])
    model.tau_w_e = np.array(parameter_model['tau_w_e'])
    model.tau_w_i = np.array(parameter_model['tau_w_i'])
    model.E_e = np.array(parameter_model['E_e'])
    model.E_i = np.array(parameter_model['E_i'])
    model.Q_e = np.array(parameter_model['Q_e'])
    model.Q_i = np.array(parameter_model['Q_i'])
    model.tau_e = np.array(parameter_model['tau_e'])
    model.tau_i = np.array(parameter_model['tau_i'])
    model.N_tot = np.array(parameter_model['N_tot'])
    model.p_connect_e = np.array(parameter_model['p_connect_e'])
    model.p_connect_i = np.array(parameter_model['p_connect_i'])
    model.g = np.array(parameter_model['g'])
    model.T = np.array(parameter_model['T'])
    model.P_e = np.array(parameter_model['P_e'])
    model.P_i = np.array(parameter_model['P_i'])
    model.K_ext_e = np.array(parameter_model['K_ext_e'])
    model.K_ext_i = np.array(parameter_model['K_ext_i'])
    model.state_variable_range['E'] = np.array(parameter_model['initial_condition']['E'])
    model.state_variable_range['I'] = np.array(parameter_model['initial_condition']['I'])
    if parameter_model['order'] == 2:
        model.state_variable_range['C_ee'] = np.array(parameter_model['initial_condition']['C_ee'])
        model.state_variable_range['C_ei'] = np.array(parameter_model['initial_condition']['C_ei'])
        model.state_variable_range['C_ii'] = np.array(parameter_model['initial_condition']['C_ii'])
    model.state_variable_range['W_e'] = np.array(parameter_model['initial_condition']['W_e'])
    model.state_variable_range['W_i'] = np.array(parameter_model['initial_condition']['W_i'])
    model.state_variable_range['ou_drift'] = np.array(parameter_model['initial_condition']['noise'])
    model.state_variable_range['external_input_excitatory_to_excitatory'] = np.array(parameter_model['initial_condition']['external_input_excitatory_to_excitatory'])
    model.state_variable_range['external_input_excitatory_to_inhibitory'] = np.array(parameter_model['initial_condition']['external_input_excitatory_to_inhibitory'])
    model.state_variable_range['external_input_inhibitory_to_excitatory'] = np.array(parameter_model['initial_condition']['external_input_inhibitory_to_excitatory'])
    model.state_variable_range['external_input_inhibitory_to_inhibitory'] = np.array(parameter_model['initial_condition']['external_input_inhibitory_to_inhibitory'])
    model.tau_OU = np.array(parameter_model['tau_OU'])
    model.weight_noise = np.array(parameter_model['weight_noise'])
    model.S_i = np.array(parameter_model['S_i'])

    connection = lab.connectivity.Connectivity(
        number_of_regions=parameter_connection_between_region['number_of_regions'],
        tract_lengths=np.ones((parameter_connection_between_region['number_of_regions'], parameter_connection_between_region['number_of_regions'])),
        weights=np.zeros((parameter_connection_between_region['number_of_regions'], parameter_connection_between_region['number_of_regions'])),
        region_labels=np.array(np.arange(0, parameter_connection_between_region['number_of_regions'], 1), dtype='U128'),
        # TODO need to replace by parameter
        centres=np.arange(0, parameter_connection_between_region['number_of_regions'], 1),
        # TODO need to replace by parameter
    )

    ## Stimulus: added by TA and Jen
    if parameter_stimulation is None:
        stimulation = None
    else:
        from tvb.basic.neotraits.api import Attr, Final
        class Sinusoid(lab.equations.TemporalApplicableEquation):
            equation = Final(
                label="Sinusoid Equation",
                default="(amp*frequency* 6.283185307179586)*cos(frequency * var * 6.283185307179586) ",
                doc=""":math:`derivate amp*sin(frequency * x * 2pi)` """)
            parameters = Attr(
                field_type=dict,
                label="Sinusoid Parameters",
                default=lambda: {"amp": 1.0, "frequency": 0.01}) #kHz #"pi": numpy.pi,

        class StimuliRegion(lab.patterns.StimuliRegion):
                def configure_time(self, time):
                    """
                    Stores the time vector, physical units (ms), as an attribute of the
                    spatio-temporal pattern and uses it to generate the temporal pattern
                    vector.
                    """
                    self.time = time
                    # Generate a discrete representation of the temporal pattern.
                    self._temporal_pattern = self.temporal.evaluate(self.time).swapaxes(0, 1).reshape(1, self.time.shape[1], -1)
                def __call__(self, temporal_indices=None, spatial_indices=None):
                    return self._temporal_pattern[0, temporal_indices]

        eqn_t = Sinusoid()
        eqn_t.parameters["amp"] = np.array(parameter_stimulation["amp"]).reshape((len(parameter_stimulation["amp"]), 1))  # ms
        eqn_t.parameters["frequency"] = np.array(parameter_stimulation["frequency"])  # ms
        stimulation = StimuliRegion(temporal=eqn_t,
                                     connectivity=connection,
                                     weight=np.array(parameter_stimulation['weights']))
        model.stvar = parameter_stimulation['variables']
    ## end add

    ## Coupling
    if parameter_coupling['type'] == 'Linear':
        coupling = lab.coupling.Linear(a=np.array(parameter_coupling['parameter']['a']),
                                       b=np.array(parameter_coupling['parameter']['b']))
    elif parameter_coupling['type'] == 'Scaling':
        coupling = lab.coupling.Scaling(a=np.array(parameter_coupling['parameter']['a']))
    elif parameter_coupling['type'] == 'HyperbolicTangent':
        coupling = lab.coupling.HyperbolicTangent(a=np.array(parameter_coupling['parameter']['a']),
                                                  b=np.array(parameter_coupling['parameter']['b']),
                                                  midpoint=np.array(parameter_coupling['parameter']['midpoint']),
                                                  sigma=np.array(parameter_coupling['parameter']['sigma']), )
    elif parameter_coupling['type'] == 'Sigmoidal':
        coupling = lab.coupling.Sigmoidal(a=np.array(parameter_coupling['parameter']['a']), b=parameter_coupling['b'],
                                          midpoint=np.array(parameter_coupling['parameter']['midpoint']),
                                          sigma=np.array(parameter_coupling['parameter']['sigma']),
                                          cmin=np.array(parameter_coupling['parameter']['cmin']),
                                          cmax=np.array(parameter_coupling['parameter']['cmax']))
    elif parameter_coupling['type'] == 'SigmoidalJansenRit':
        coupling = lab.coupling.SigmoidalJansenRit(a=np.array(parameter_coupling['parameter']['a']),
                                                   b=parameter_coupling['b'],
                                                   midpoint=np.array(parameter_coupling['parameter']['midpoint']),
                                                   r=np.array(parameter_coupling['parameter']['r']),
                                                   cmin=np.array(parameter_coupling['parameter']['cmin']),
                                                   cmax=np.array(parameter_coupling['parameter']['cmax']))
    elif parameter_coupling['type'] == 'PreSigmoidal':
        coupling = lab.coupling.PreSigmoidal(H=np.array(parameter_coupling['parameter']['H']),
                                             b=parameter_coupling['b'],
                                             Q=np.array(parameter_coupling['parameter']['Q']),
                                             G=np.array(parameter_coupling['parameter']['G']),
                                             P=np.array(parameter_coupling['parameter']['P']),
                                             theta=np.array(parameter_coupling['parameter']['theta']),
                                             dynamic=np.array(parameter_coupling['parameter']['dynamic']),
                                             globalT=np.array(parameter_coupling['parameter']['globalT']),
                                             )
    elif parameter_coupling['type'] == 'Difference':
        coupling = lab.coupling.Difference(a=np.array(parameter_coupling['parameter']['a']))
    elif parameter_coupling['type'] == 'Kuramoto':
        coupling = lab.coupling.Kuramoto(a=np.array(parameter_coupling['parameter']['a']))
    else:
        raise Exception('Bad type for the coupling')

    ## Integrator
    if not parameter_integrator['stochastic']:
        if parameter_integrator['type'] == 'Heun':
            integrator = lab.integrators.HeunDeterministic(dt=parameter_integrator['dt'])
        elif parameter_integrator['type'] == 'Euler':
            integrator = lab.integrators.EulerDeterministic(dt=parameter_integrator['dt'])
        elif parameter_integrator['type'] == 'Indentity':
            integrator = lab.integrators.Identity(dt=parameter_integrator['dt'])
        elif parameter_integrator['type'] == 'RungeKutta4th':
            integrator = lab.integrators.RungeKutta4thOrderDeterministic(dt=parameter_integrator['dt'])
        else:
            raise Exception('Bad type for the integrator')
    else:
        if parameter_integrator['noise_type'] == 'Additive':
            noise = lab.noise.Additive(nsig=np.array(parameter_integrator['noise_parameter']['nsig']),
                                       ntau=parameter_integrator['noise_parameter']['ntau'], )
        else:
            raise Exception('Bad type for the noise')
        noise.random_stream.seed(parameter_simulation['seed'])

        if parameter_integrator['type'] == 'Heun':
            integrator = lab.integrators.HeunStochastic(noise=noise, dt=parameter_integrator['dt'])
        elif parameter_integrator['type'] == 'Euler':
            integrator = lab.integrators.EulerStochastic(noise=noise, dt=parameter_integrator['dt'])
        elif parameter_integrator['type'] == 'Indentity':
            integrator = lab.integrators.IdentityStochastic(noise=noise, dt=parameter_integrator['dt'])
        else:
            raise Exception('Bad type for the integrator')

    ## Monitors
    monitors = []
    if parameter_monitor['Raw']:
        monitors.append(lab.monitors.Raw())
    if parameter_monitor['TemporalAverage']:
        monitor_TAVG = lab.monitors.TemporalAverage(
            variables_of_interest=np.array(parameter_monitor['parameter_TemporalAverage']['variables_of_interest']),
            period=parameter_monitor['parameter_TemporalAverage']['period'])
        monitors.append(monitor_TAVG)
    if parameter_monitor['Bold']:
        monitor_Bold = lab.monitors.Bold(
            variables_of_interest=parameter_monitor['parameter_Bold']['variables_of_interest'],
            period=parameter_monitor['parameter_Bold']['period'])
        monitors.append(monitor_Bold)
    if 'Afferent_coupling' in parameter_monitor.keys() and parameter_monitor['Afferent_coupling']:
        monitor_Afferent_coupling = lab.monitors.AfferentCoupling(variables_of_interest=None)
        monitors.append(monitor_Afferent_coupling)

    # save the parameters in on file
    if not os.path.exists(parameter_simulation['path_result']):
        os.mkdir(parameter_simulation['path_result'])
    f = open(parameter_simulation['path_result'] + '/parameter.json', "w")
    f.write("{\n")
    for name, dic in [('parameter_simulation', parameter_simulation),
                      ('parameter_model', parameter_model),
                      ('parameter_connection_between_region', parameter_connection_between_region),
                      ('parameter_coupling', parameter_coupling),
                      ('parameter_integrator', parameter_integrator),
                      ('parameter_monitor', parameter_monitor)]:
        f.write('"' + name + '" : ')
        json.dump(dic, f)
        f.write(",\n")
    if parameter_stimulation is not None:
        f.write('"parameter_stimulation" : ')
        json.dump(parameter_stimulation, f)
        f.write(",\n")

    f.write('"myseed":' + str(my_seed) + "\n}\n")
    f.close()

    # initialize the simulator: edited by TA and Jen, added stimulation argument, try removing surface
    if initial_condition == None:
        simulator = lab.simulator.Simulator(model=model, connectivity=connection,
                                            coupling=coupling, integrator=integrator, monitors=monitors,
                                            stimulus=stimulation)
    else:
        simulator = lab.simulator.Simulator(model=model, connectivity=connection,
                                            coupling=coupling, integrator=integrator,
                                            monitors=monitors, initial_conditions=initial_condition,
                                            stimulus=stimulation)
    simulator.configure()
    if initial_condition == None:
        # save the initial condition
        np.save(parameter_simulation['path_result'] + '/step_init.npy', simulator.history.buffer)
        # end edit
    return simulator


def run_simulation(simulator, time, parameter_simulation, parameter_monitor):
    """
    run a simulation
    :param simulator: the simulator already initialize
    :param time: the time of simulation
    :param parameter_simulation: the parameter for the simulation
    :param parameter_monitor: the parameter for the monitor
    """
    # check how many monitor it's used
    nb_monitor = parameter_monitor['Raw'] + parameter_monitor['TemporalAverage'] + parameter_monitor['Bold']
    if 'Afferent_coupling' in parameter_monitor.keys() and parameter_monitor['Afferent_coupling']:
        nb_monitor += 1
    # initialise the variable for the saving the result
    save_result = []
    for i in range(nb_monitor):
        save_result.append([])
    # run the simulation
    count = 0
    for result in simulator(simulation_length=time):
        for i in range(nb_monitor):
            if result[i] is not None:
                save_result[i].append(result[i])
        # save the result in file
        if result[0][0] >= parameter_simulation['save_time'] * (
                count + 1):  # check if the time for saving at some time step
            print('simulation time :' + str(result[0][0]) + '\r')
            np.save(parameter_simulation['path_result'] + '/step_' + str(count) + '.npy', np.array(save_result, dtype=object))
            save_result = []
            for i in range(nb_monitor):
                save_result.append([])
            count += 1
    # save the last part
    np.save(parameter_simulation['path_result'] + '/step_' + str(count) + '.npy', np.array(save_result, dtype=object))


def get_result(path, time_begin, time_end):
    """
    return the result of the simulation between the wanted time
    :param path: the folder of the simulation
    :param time_begin: the start time for the result
    :param time_end:  the ending time for the result
    :return: result of all monitor
    """
    with open(path + '/parameter.json') as f:
        parameters = json.load(f)
    parameter_simulation = parameters['parameter_simulation']
    parameter_monitor = parameters['parameter_monitor']
    count_begin = int(time_begin / parameter_simulation['save_time'])
    count_end = int(time_end / parameter_simulation['save_time']) + 1
    nb_monitor = parameter_monitor['Raw'] + parameter_monitor['TemporalAverage'] + parameter_monitor['Bold']
    if 'Afferent_coupling' in parameter_monitor.keys() and parameter_monitor['Afferent_coupling']:
        nb_monitor += 1
    output = []

    for count in range(count_begin, count_end):
        result = np.load(path + '/step_' + str(count) + '.npy', allow_pickle=True)
        for i in range(result.shape[0]):
            tmp = np.array(result[i])
            if len(tmp) != 0:
                tmp = tmp[np.where((time_begin <= tmp[:, 0]) & (tmp[:, 0] <= time_end)), :]
                tmp_time = tmp[0][:, 0]
                if tmp_time.shape[0] != 0:
                    one = tmp[0][:, 1][0]
                    tmp_value = np.concatenate(tmp[0][:, 1]).reshape(tmp_time.shape[0], one.shape[0], one.shape[1])
                    if len(output) == nb_monitor:
                        output[i] = [np.concatenate([output[i][0], tmp_time]),
                                     np.concatenate([output[i][1], tmp_value])]
                    else:
                        output.append([tmp_time, tmp_value])
    return output


def get_region(path, time_begin, time_end, region_nb):
    """
    return the result of the simulation between the wanted time
    :param path: the folder of the simulation
    :param time_begin: the start time for the result
    :param time_end: the ending time for the result
    :param region_nb: interger of array of interger for the select the region of different regions
    :return: result of all monitor
    """
    with open(path + '/parameter.json') as f:
        parameters = json.load(f)
    parameter_simulation = parameters['parameter_simulation']
    parameter_monitor = parameters['parameter_monitor']
    count_begin = int(time_begin / parameter_simulation['save_time'])
    count_end = int(time_end / parameter_simulation['save_time']) + 1
    nb_monitor = parameter_monitor['Raw'] + parameter_monitor['TemporalAverage'] + parameter_monitor['Bold']
    output = []

    for count in range(count_begin, count_end):
        result = np.load(path + '/step_' + str(count) + '.npy')
        for i in range(result.shape[0]):
            tmp = np.array(result[i])
            if len(tmp) != 0:
                tmp = tmp[np.where((time_begin <= tmp[:, 0]) & (tmp[:, 0] <= time_end)), :]
                tmp_time = tmp[0][:, 0]
                if tmp_time.shape[0] != 0:
                    one = tmp[0][:, 1][0]
                    tmp_value = np.concatenate(tmp[0][:, 1]).reshape(tmp_time.shape[0], one.shape[0], one.shape[1])
                    tmp_value = tmp_value[:, :, region_nb]
                    if len(output) == nb_monitor:
                        output[i] = [np.concatenate([output[i][0], tmp_time]),
                                     np.concatenate([output[i][1], tmp_value])]
                    else:
                        output.append([tmp_time, tmp_value])
    return output


def print_all_activity(path, time_begin, time_end, position_monitor, position_variable):
    """
    print one value of on monitor for some times
    :param path: the folder of the simulation
    :param time_begin: the start time for the result
    :param time_end: the ending time for the result
    :param position_monitor: select the monitor
    :param position_variable: select the variable of monitor
    :return: nothing
    """
    output = get_result(path, time_begin, time_end)
    import matplotlib.pyplot as plt
    plt.plot(output[position_monitor][0] * 1e-3,  # time in second
             output[position_monitor][1][:, position_variable, :]
             )
    plt.show()


def print_EI_one(path, time_begin, time_end, position_monitor, position_node):
    """
    plot on region with excitatory and inhibitory
    :param path: the folder of the simulation
    :param time_begin: the start time for the result
    :param time_end: the ending time for the result
    :param position_monitor: select the monitor
    :param position_node:
    :return: nothing
    """
    output = get_result(path, time_begin, time_end)
    import matplotlib.pyplot as plt
    plt.plot(output[position_monitor][0] * 1e-3,  # time in second
             output[position_monitor][1][:, 0, position_node] * 1e3,
             color='c')
    plt.plot(output[position_monitor][0] * 1e-3,  # time in second
             output[position_monitor][1][:, 1, position_node] * 1e3,
             color='r')
    # plt.figure()
    # plt.plot(output[position_monitor][0]*1e-3, # time in second
    #          output[position_monitor][1][:,5,position_node],
    #          color='k')
    # plt.show()


def print_region(path, time_begin, time_end, position_monitor, position_variable, nb_region):
    """
    print one value of on monitor for some times
    :param path: the folder of the simulation
    :param time_begin: the start time for the result
    :param time_end: the ending time for the result
    :param position_monitor: select the monitor
    :param position_variable: select the variable of monitor
    :param nb_region: interger of array of interger for the select the region of different regions
    :return: nothing
    """
    output = get_region(path, time_begin, time_end, nb_region)
    import matplotlib.pyplot as plt
    plt.plot(output[position_monitor][0] * 1e-3,  # time in second
             output[position_monitor][1][:, position_variable, :]
             )
    plt.show()


def print_bistability(parameter_model, show=True):
    """
    print if the model is bistable or not
    :param parameter_model: parameters for the model
        (the parameter external_input_in_in and external_input_in_ex is taking in count)
    :return: nothing
    """
    if parameter_model['matteo']:
        import tvb_model_reference.src.Zerlaut_matteo as model
    else:
        import tvb_model_reference.src.Zerlaut as model
    ## Model
    if parameter_model['order'] == 1:
        model = model.Zerlaut_adaptation_first_order(variables_of_interest='E I W_e W_i'.split())
    elif parameter_model['order'] == 2:
        model = model.Zerlaut_adaptation_second_order(variables_of_interest='E I C_ee C_ei C_ii W_e W_i'.split())
    else:
        raise Exception('Bad order for the model')

    model.g_L = np.array(parameter_model['g_L'])
    model.E_L_e = np.array(parameter_model['E_L_e'])
    model.E_L_i = np.array(parameter_model['E_L_i'])
    model.C_m = np.array(parameter_model['C_m'])
    model.b_e = np.array(parameter_model['b_e'])
    model.a_e = np.array(parameter_model['a_e'])
    model.b_i = np.array(parameter_model['b_i'])
    model.a_i = np.array(parameter_model['a_i'])
    model.tau_w_e = np.array(parameter_model['tau_w_e'])
    model.tau_w_i = np.array(parameter_model['tau_w_i'])
    model.E_e = np.array(parameter_model['E_e'])
    model.E_i = np.array(parameter_model['E_i'])
    model.Q_e = np.array(parameter_model['Q_e'])
    model.Q_i = np.array(parameter_model['Q_i'])
    model.tau_e = np.array(parameter_model['tau_e'])
    model.tau_i = np.array(parameter_model['tau_i'])
    model.N_tot = np.array(parameter_model['N_tot'])
    model.p_connect_e = np.array(parameter_model['p_connect_e'])
    model.p_connect_i = np.array(parameter_model['p_connect_i'])
    model.g = np.array(parameter_model['g'])
    model.T = np.array(parameter_model['T'])
    model.P_e = np.array(parameter_model['P_e'])
    model.P_i = np.array(parameter_model['P_i'])
    model.K_ext_e = np.array(parameter_model['K_ext_e'])
    model.K_ext_i = np.array(parameter_model['K_ext_i'])
    model.external_input_ex_ex = np.array(parameter_model['external_input_ex_ex'])
    model.external_input_ex_in = np.array(parameter_model['external_input_ex_in'])
    model.external_input_in_ex = np.array(parameter_model['external_input_in_ex'])
    model.external_input_in_in = np.array(parameter_model['external_input_in_in'])
    model.tau_OU = np.array(parameter_model['tau_OU'])
    model.weight_noise = np.array(parameter_model['weight_noise'])
    model.state_variable_range['E'] = np.array(parameter_model['initial_condition']['E'])
    model.state_variable_range['I'] = np.array(parameter_model['initial_condition']['I'])
    if parameter_model['order'] == 2:
        model.state_variable_range['C_ee'] = np.array(parameter_model['initial_condition']['C_ee'])
        model.state_variable_range['C_ei'] = np.array(parameter_model['initial_condition']['C_ei'])
        model.state_variable_range['C_ii'] = np.array(parameter_model['initial_condition']['C_ii'])
    model.state_variable_range['W_e'] = np.array(parameter_model['initial_condition']['W_e'])
    model.state_variable_range['W_i'] = np.array(parameter_model['initial_condition']['W_i'])

    ##Solution equation
    def equation(x):
        return (model.TF_inhibitory((fezero + model.external_input_ex_in), x, 0.0, 0.0, 0.0) - x)
        # return (model.TF_inhibitory((fezero+extinp), x,fezero*model.b_e*model.tau_w_e)-x) # with fix adaptation

    # raneg fo frequency in KHz
    xrange1 = np.arange(0.0, 200.0, 0.01) * 1e-3
    feprimovec = 0 * xrange1
    fiprimovec = 0 * xrange1
    for i in range(len(xrange1)):
        fezero = xrange1[i]
        fizero = fsolve(equation, (fezero * 2), xtol=1.e-35)
        fezeroprime = model.TF_excitatory(fezero + model.external_input_ex_ex, fizero, 0.0, 0.0, 0.0)
        # fezeroprime=model.TF_excitatory(fezero, fizero,fezero*model.b_e*model.tau_w_e) # with fixe adaptation
        feprimovec[i] = fezeroprime
        fiprimovec[i] = fizero
        # print(i,np.array(xrange1[i]),fezeroprime,fizero)
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(xrange1 * 1e3, feprimovec * 1e3, 'b-', xrange1 * 1e3, xrange1 * 1e3, 'k--')
    plt.plot(fiprimovec * 1e3, feprimovec * 1e3, 'r-')
    plt.xlim(0, 10)
    plt.ylim(0, 10)
    plt.savefig('Plot_Bistability.pdf')
    if show:
        plt.show()
    return xrange1, feprimovec


def print_space_variable(path, time_begin, time_end, position_monitor, limit=True):
    """
    plot the space variable of a result
    :param path: path
    :param time_begin: start of the plot
    :param time_end: end of the plot
    :param position_monitor: monitor selected
    :param limit: use define limit
    :return:
    """
    output = get_result(path, time_begin, time_end)
    with open(path + '/parameter.json') as f:
        parameters = json.load(f)
    simulator = init(parameters['parameter_simulation'],
                     parameters['parameter_model'],
                     parameters['parameter_connection_between_region'],
                     parameters['parameter_coupling'],
                     parameters['parameter_integrator'],
                     parameters['parameter_monitor'])

    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    for region in range(output[position_monitor][1].shape[2]):
        delays = simulator.history.delays[region, :]
        weight = simulator.history.weights[region, :]
        initial_condition = np.load(path + '/step_init.npy')
        output_init = np.concatenate((initial_condition[:, 0, :, 0], output[0][1][:, 0, :]))
        for index, delay in enumerate(delays):
            output_init[:, index] = np.concatenate((np.zeros(int(delay) + 1), output_init[:-int(delay) - 1, index]))
        max_time = len(output[position_monitor][1][:, 0, region])
        External_input = np.ones((max_time, 1))
        for t in range(max_time):
            tmp = np.sum(output_init[-max_time + t] * weight)
            External_input[t] = simulator.coupling.post(tmp)
        fig = plt.figure(figsize=(20, 8))
        gs1 = gridspec.GridSpec(3, 3, figure=fig)
        axs = np.array(
            [[plt.subplot(gs1[0, 0]), plt.subplot(gs1[0, 1])], [plt.subplot(gs1[1, 0]), plt.subplot(gs1[1, 1])]])

        plt.suptitle('Analysis of region' + str(region), fontsize=16)
        axs[0, 0].set_title('Firing rate excitatory vs firing rate inhibitory')
        axs[0, 0].plot(
            output[position_monitor][1][:, 1, region],
            output[position_monitor][1][:, 0, region]
        )
        axs[0, 0].set_ylabel('Firing rate excitatory')
        axs[0, 0].set_xlabel('Firing rate inhibitory')
        if limit:
            axs[0, 0].set_ylim(ymin=-0.001, ymax=0.02)
            axs[0, 0].set_xlim(xmin=-0.001, xmax=0.03)

        axs[0, 1].plot(
            output[position_monitor][1][:, 5, region],
            output[position_monitor][1][:, 0, region],
        )
        axs[0, 1].set_ylabel('Firing rate excitatory')
        axs[0, 1].set_xlabel('Adaptation')
        axs[0, 1].set_title('Firing rate excitatory vs Adaptation', fontsize=16)
        if limit:
            axs[0, 1].set_ylim(ymin=-0.001, ymax=0.02)
            axs[0, 1].set_xlim(xmin=-1.0, xmax=120.0)

        nb_time = len(output[position_monitor][1][:, 0, region])
        axs[1, 0].plot(
            External_input,
            output[position_monitor][1][:, 0, region],
        )
        axs[1, 0].set_ylabel('Firing rate excitatory')
        axs[1, 0].set_xlabel('External input')
        axs[1, 0].set_title('Firing rate excitatory vs External input', fontsize=16)
        if limit:
            axs[1, 0].set_ylim(ymin=-0.001, ymax=0.02)
            axs[1, 0].set_xlim(xmin=0.00, xmax=0.008)

        axs[1, 1].plot(
            output[position_monitor][1][:, 5, region],
            External_input
        )
        axs[1, 1].set_xlabel('Adaptation')
        axs[1, 1].set_ylabel('External input')
        axs[1, 1].set_title('Adaptation vs External input', fontsize=16)
        if limit:
            axs[1, 1].set_ylim(ymin=0.0, ymax=0.008)
            axs[1, 1].set_xlim(xmin=-1.0, xmax=120.0)

        ax_time = plt.subplot(gs1[2, :])
        ax_time.plot(
            output[position_monitor][0],
            output[position_monitor][1][:, 0, region],
        )
        ax_time.plot(
            output[position_monitor][0],
            output[position_monitor][1][:, 1, region],
        )
        ax_time.set_xlabel('time')
        ax_time.set_ylabel('Firing rate')
        ax_time.set_title('time', fontsize=16)
        if limit:
            ax_time.set_ylim(ymin=0.0, ymax=0.03)

        from mpl_toolkits.mplot3d import Axes3D
        axbig = plt.subplot(gs1[:2, 2], projection='3d')
        axbig.plot(output[position_monitor][1][:, 1, region],
                   output[position_monitor][1][:, 0, region],
                   output[position_monitor][1][:, 5, region],
                   )
        axbig.set_ylabel('Firing rate excitatory')
        axbig.set_xlabel('Firing rate inhibitory')
        axbig.set_zlabel('Adapation')
        if limit:
            axbig.set_ylim(ymin=-0.001, ymax=0.02)
            axbig.set_xlim(xmin=-0.001, xmax=0.03)
            axbig.set_zlim(zmin=-1.0, zmax=120.0)

        plt.savefig(path + '/region' + str(region) + '.png')
        plt.close('all')


def print_all(path, time_begin, time_end, position_monitor, con, size=0.031, shift=0.0, E_I=True, title=None,
              drop_mean=False):
    """
    print one value of on monitor for some times
    :param path: the folder of the simulation
    :param time_begin: the start time for the result
    :param time_end: the ending time for the result
    :param position_monitor: select the monitor
    :param con: connectivity
    :param size: size of plotting
    :param shift: size of the shift
    :param E_I: E and I plot or not
    :param title: title of the figure
    :return: nothing
    """
    output = get_result(path, time_begin, time_end)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(20, 20))
    nb_region = output[position_monitor][1].shape[2]
    for i in range(nb_region):
        if drop_mean:
            mean = np.mean(output[position_monitor][1][:, 0, i])
        else:
            mean = 0.0
        if E_I:
            plt.plot(output[position_monitor][0] * 1e-3,  # time in second
                     output[position_monitor][1][:, 0, i] + size * i + shift,
                     linewidth=0.5,
                     color='r'
                     )
            plt.plot(output[position_monitor][0] * 1e-3,  # time in second
                     output[position_monitor][1][:, 1, i] + size * i + shift,
                     color='b',
                     linewidth=0.5,
                     )
        else:
            plt.plot(output[position_monitor][0] * 1e-3,  # time in second
                     output[position_monitor][1][:, 0, i] + size * i + shift - mean,
                     color='b',
                     linewidth=0.5,
                     )
    plt.xlim(xmax=time_end * 1e-3 + 0.1, xmin=time_begin * 1e-3 - 0.1)
    plt.ylim(ymin=-size / 2, ymax=nb_region * size)
    plt.xlabel('time in s', {'fontsize': 20})
    plt.ylabel('regions', {'fontsize': 20})
    plt.yticks(np.arange(0, nb_region, 1) * size, con.region_labels)
    if title is not None:
        plt.title(title)
    plt.show()

