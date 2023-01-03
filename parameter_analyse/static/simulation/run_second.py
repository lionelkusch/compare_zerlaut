from parameter_analyse.static.python_file.parameters import parameter_default
from parameter_analyse.static.python_file.run.run_exploration import save_parameter, generate_parameter
from parameter_analyse.static.python_file.simulation.simulation_time_evolve import simulate
import numpy as np
import os
import datetime


def run_sim(results_path, parameter_default, dict_variable, duration, max_step, nb_transient):
    """
    Run one simulation, analyse the simulation and save this result in the database
    simulation where the external firing reduce each step of specific duration
    :param results_path: the folder where to save spikes
    :param data_base: the file of the database
    :param table_name: the name of the table of the database
    :param dict_variable : dictionary with the variable change
    :param begin: the beginning of record spike
    :param end: the end of the simulation
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
             max_step=max_step, nb_transient=nb_transient
             )

    print('time: ' + str(datetime.datetime.now()) + ' END SIMULATION \n')

if __name__ == '__main__':
    for b in [0.0, 30., 60.]:
        path = '/home/kusch/Documents/project/Zerlaut/compare_zerlaut/parameter_analyse/static/simulation/time_reduce/b_'+str(b)+'/'
        parameter_default.param_nest['local_num_threads'] = 8
        parameter_default.param_topology['mean_w_0'] = 200.0 if b != 0.0 else 0.0
        run_sim(path, parameter_default, {'b': b, 'rate': 52.0}, 10000.0, 53, 1)
