#  Copyright 2023 Aix-Marseille Université
# "Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements; and to You under the Apache License, Version 2.0. "
import os
import datetime
from parameter_analyse.static.python_file.parameters import parameter_default
from parameter_analyse.static.python_file.run.run_exploration import save_parameter, generate_parameter
from parameter_analyse.static.python_file.simulation.simulation_time_evolve import simulate


def run_sim(results_path, parameter_default, dict_variable, duration, max_step, extra=0):
    """
    Run one simulation, analyse the simulation and save this result in the database
    simulation where the external firing reduce each step of specific duration
    :param results_path: the folder where to save spikes
    :param parameter_default: default parameters for simulation
    :param dict_variable : dictionary with the variable change
    :param duration: duration of each step
    :param max_step: maximum of step
    :param extra: extra step
    :return: nothing
    """
    print('time: ' + str(datetime.datetime.now()) + ' BEGIN SIMULATION \n')
    # create the folder for result is not exist
    newpath = os.path.join(os.getcwd(), results_path)
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    elif os.path.exists(newpath + '/spike_recorder_ex.dat'):
        print('Simulation already done ')
        print('time: ' + str(datetime.datetime.now()) + ' END SIMULATION \n')
        return

    param_nest, param_topology, param_connexion, param_background = generate_parameter(parameter_default, dict_variable)

    save_parameter({"param_nest": param_nest, "param_topology": param_topology,
                    "param_connexion": param_connexion, "param_background": param_background},
                   results_path)

    # simulate
    simulate(results_path=results_path, duration=duration,
             param_nest=param_nest, param_topology=param_topology,
             param_connexion=param_connexion, param_background=param_background,
             max_step=max_step, extra=extra
             )

    print('time: ' + str(datetime.datetime.now()) + ' END SIMULATION \n')


if __name__ == '__main__':
    for b in [0.0, 30., 60.]:
        path = os.path.dirname(os.path.realpath(__file__)) + '/data/time_reduce/b_'+str(b)+'/'
        parameter_default.param_nest['local_num_threads'] = 8
        parameter_default.param_topology['mean_w_0'] = 200.0 if b != 0.0 else 0.0
        run_sim(path, parameter_default, {'b': b, 'rate': 52.0}, 10000.0, 53, extra=0)
