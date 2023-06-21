#  Copyright 2023 Aix-Marseille Université
# "Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements; and to You under the Apache License, Version 2.0. "
import sqlite3
import datetime
import os
import sys
import numpy as np
import pathos.multiprocessing as mp
import dill
import json

from parameter_analyse.static.python_file.simulation.simulation import simulate
from parameter_analyse.static.python_file.analysis.analysis_global import analysis_global
from parameter_analyse.static.python_file.analysis.result_class import Result_analyse


def generate_parameter(parameter_default, dict_variable):
    """
    generate the parameters for the simulation
    :param parameter_default: parameters by default
    :param dict_variable: dictionary with the parameters to change
    :return:
    """
    param_nest = parameter_default.param_nest
    param_topology = parameter_default.param_topology
    param_connexion = parameter_default.param_connexion
    param_background = parameter_default.param_background
    for variable in dict_variable.keys():
        if variable in param_nest.keys():
            param_nest[variable] = dict_variable[variable]
        elif variable in param_topology.keys():
            param_topology[variable] = dict_variable[variable]
        elif variable in param_topology['excitatory_param'].keys():  # only excitatory neurons can have modification
            param_topology['excitatory_param'][variable] = dict_variable[variable]
        elif variable in param_connexion.keys():
            param_connexion[variable] = dict_variable[variable]
        elif variable in param_background.keys():
            param_background[variable] = dict_variable[variable]
    return param_nest, param_topology, param_connexion, param_background


def get_resolution(parameter_default, dict_variable):
    """
    change the duration of the simulation
    :param parameter_default:
    :param dict_variable:
    :return:
    """
    if 'sim_resolution' in dict_variable.keys():
        return dict_variable['sim_resolution']
    else:
        return parameter_default.param_nest['sim_resolution']


def save_parameter(parameters, result_path):
    """
    save the parameters of the simulations in json file
    :param parameters: dictionary of parameters
    :param result_path: path of the result
    :return: nothing
    """
    # save the value of all parameters
    f = open(result_path + '/parameter.json', "wt")
    json.dump(parameters, f)
    f.close()


def type_database(variable):
    """
    type of the variable for saving in database
    :param variable:
    :return:
    """
    if hasattr(variable, 'dtype'):
        if np.issubdtype(variable, int):
            return 'INTEGER'
        elif np.issubdtype(variable, float):
            return 'REAL'
        else:
            sys.stderr.write('ERROR bad type of save variable\n')
            exit(1)
    else:
        if isinstance(variable, int):
            return 'INTEGER'
        elif isinstance(variable, float):
            return 'REAL'
        elif isinstance(variable, str):
            return 'TEXT'
        else:
            sys.stderr.write('ERROR bad type of save variable\n')
            exit(1)


def init_database(data_base, table_name, dict_variable):
    """
    Initialise the connection to the database et create the table
    :param data_base: file where is the database
    :param table_name: the name of the table
    :param dict_variable: dictionary with the parameters to change
    :return: the connexion to the database
    """
    variable = ''
    key_variable = ','
    for key in dict_variable.keys():
        variable += key + ' ' + type_database(dict_variable[key]) + ' NOT NULL,'
        key_variable += key + ','
    measures_name = list(Result_analyse().name_measure())
    measures_name.remove('names_population')
    measures = ''
    for name in measures_name:
        measures += name + ' REAL,'

    con = sqlite3.connect(data_base, detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES, timeout=10000)
    cur = con.cursor()
    cur.execute('CREATE TABLE IF NOT EXISTS '
                + table_name
                + '(date TIMESTAMP NOT NULL,'
                  'path_file TEXT NOT NULL,'
                + variable
                + 'names_population TEXT NOT NULL,'
                + measures
                + 'PRIMARY KEY'
                  '(path_file' + key_variable + 'names_population))'
                )
    cur.close()
    con.close()


def check_already_analise_database(data_base, table_name, result_path, name_population):
    """
    Check if the analysis was already perform
    :param data_base: path of the database
    :param table_name: name of the table
    :param result_path: folder to analyse
    :param name_population: name of the population to analise
    """
    con = sqlite3.connect(data_base, detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES)
    cursor = con.cursor()
    cursor.execute("SELECT * FROM "+table_name+" WHERE path_file = '"+result_path+"' AND names_population='"+name_population+"'")
    check = len(cursor.fetchall()) != 0
    cursor.close()
    con.close()
    return check


def insert_database(data_base, table_name, results_path, dict_variable, result):
    """
    Insert some result in the database
    :param data_base: file of the database
    :param table_name: the table where insert the value
    :param results_path: the path where is the results
    :param dict_variable: dictionary with the parameter to save
    :param result: the measure of the networks
    :return: nothing
    """
    con = sqlite3.connect(data_base, detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES, timeout=1000)
    cur = con.cursor()
    list_data = []
    for pop in range(len(result['names_population'])):
        data = {'date': datetime.datetime.now(),
                'path_file': results_path}
        data.update(dict_variable)
        for key in result.keys():
            data[key] = result[key][pop]
        list_data.append(tuple(data.values()))
    keys = ','.join(data.keys())
    question_marks = ','.join(list('?' * len(data)))
    cur.executemany('INSERT INTO ' + table_name + ' (' + keys + ') VALUES (' + question_marks + ')', list_data)
    con.commit()
    cur.close()
    con.close()


def run(results_path, parameter_default, dict_variable, begin, end):
    """
    Run one simulation, analyse the simulation and save this result in the database
    :param results_path: the folder where to save spikes
    :param parameter_default: parameter by default
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
    elif os.path.exists(newpath+'/spike_recorder_ex.dat'):
        print('Simulation already done ')
        print('time: ' + str(datetime.datetime.now()) + ' END SIMULATION \n')
        return

    param_nest, param_topology, param_connexion, param_background = generate_parameter(parameter_default, dict_variable)

    save_parameter({"param_nest":param_nest, "param_topology":param_topology,
     "param_connexion":param_connexion, "param_background":param_background},
                   results_path)

    # simulate
    simulate(results_path=results_path, begin=begin, end=end,
             param_nest=param_nest, param_topology=param_topology,
             param_connexion=param_connexion, param_background=param_background
             )

    print('time: ' + str(datetime.datetime.now()) + ' END SIMULATION \n')


def analysis(results_path, parameter_default, data_base, table_name, dict_variable, begin, end):
    """
    analysis the result
    :param results_path: path of the result simulation
    :param parameter_default: parameter by default
    :param data_base: file of database
    :param table_name: table for saving
    :param dict_variable: dictionary for parameters which change
    :param begin: start analysis
    :param end: end analysis
    :return:
    """
    print('time: ' + str(datetime.datetime.now()) + ' BEGIN ANALYSIS \n')
    # Save analysis in database
    init_database(data_base, table_name, dict_variable)

    # check if the analysis was already perform
    if check_already_analise_database(data_base, table_name, results_path, "excitatory"):
        print('Analysis already done excitatory')
        print('time: ' + str(datetime.datetime.now()) + ' END ANALYSIS "excitatory"\n')
    else:
        # excitatory analysis
        result_global = analysis_global(path=results_path, number=0, begin=begin, end=end,
                                        resolution=get_resolution(parameter_default, dict_variable),
                                        limit_burst=10.0)
        # try:
        insert_database(data_base, table_name, results_path, dict_variable, result_global)
        # except sqlite3.IntegrityError:
        #     exit(0)
        print('time: ' + str(datetime.datetime.now()) + ' END ANALYSIS excitatory \n')
    if check_already_analise_database(data_base, table_name, results_path, "inhibitory"):
        print('Analysis already done inhibitory')
        print('time: ' + str(datetime.datetime.now()) + ' END ANALYSIS inhibitory\n')
    else:
        # inhibitory
        result_global = analysis_global(path=results_path, number=1, begin=begin, end=end,
                                        resolution=get_resolution(parameter_default, dict_variable),
                                        limit_burst=10.0)
        # try:
        insert_database(data_base, table_name, results_path, dict_variable, result_global)
        # except sqlite3.IntegrityError:
        #     exit(0)
        print('time: ' + str(datetime.datetime.now()) + ' END ANALYSIS inhibitory\n')


def run_exploration_2D(path, parameter_default, data_base, table_name, dict_variables, begin, end,
                       analyse=True, simulation=True):
    """
    run exploration with 2 parameters to change
    :param path: path for saving result of simulation
    :param parameter_default: parameter by default ofr the simulation
    :param data_base: database for saving the result
    :param table_name: table for saving exploration
    :param dict_variables: dictionary of parameter which change
    :param begin: begin of analysis and recording
    :param end: end of simulation and analysis
    :param analyse: analysis
    :param simulation: simulate
    :return:
    """
    name_variable_1, name_variable_2 = dict_variables.keys()
    print(path)
    if simulation:
        for variable_1 in dict_variables[name_variable_1]:
            for variable_2 in dict_variables[name_variable_2]:
                # try:
                    print(
                        'SIMULATION : ' + name_variable_1 + ': ' + str(variable_1) + ' ' + name_variable_2 + ': ' + str(
                            variable_2))
                    results_path = path + '_' + name_variable_1 + '_' + str(
                        variable_1) + '_' + name_variable_2 + '_' + str(variable_2)
                    run(results_path, parameter_default, {name_variable_1: variable_1, name_variable_2: variable_2},
                        begin, end)
                # except:
                #     sys.stderr.write('time: ' + str(datetime.datetime.now()) + ' error: ERROR in simulation \n')
    if analyse:
        list_path = []
        for variable_1 in dict_variables[name_variable_1]:
            for variable_2 in dict_variables[name_variable_2]:
                # analysis(path+'_'+name_variable_1+'_'+str(variable_1)+'_'+name_variable_2+'_'+str(variable_2),parameter_default,data_base,table_name,{name_variable_1:variable_1,name_variable_2:variable_2},begin,end)
                list_path.append((path + '_' + name_variable_1 + '_' + str(
                    variable_1) + '_' + name_variable_2 + '_' + str(variable_2), name_variable_1, variable_1,
                                  name_variable_2, variable_2))
        # multithreading analyse
        p = mp.ProcessingPool(ncpus=parameter_default.param_nest['local_num_threads'])

        def analyse_function(arg):
            path, name_variable_1, variable_1, name_variable_2, variable_2 = arg
            # try:
            print('ANALYSIS : ' + name_variable_1 + ': ' + str(variable_1) + ' ' + name_variable_2 + ': ' + str(
                variable_2) + '\n')
            print(path, parameter_default, data_base, table_name,
                  {name_variable_1: variable_1, name_variable_2: variable_2}, begin, end,)
            analysis(path, parameter_default, data_base, table_name,
                     {name_variable_1: variable_1, name_variable_2: variable_2}, begin, end,
                     )
            # except:
            #     print('ANALYSIS : ' + name_variable_1 + ': ' + str(variable_1) + ' ' + name_variable_2 + ': ' + str(
            #         variable_2))
            #     sys.stderr.write('time: ' + str(datetime.datetime.now()) + ' error: ERROR in analyse \n')

        p.map(dill.copy(analyse_function), list_path)
