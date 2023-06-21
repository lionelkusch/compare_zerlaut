#  Copyright 2023 Aix-Marseille Université
# "Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements; and to You under the Apache License, Version 2.0. "
param_nest = {
    # Resolution of the simulation (in ms).
    'sim_resolution': 0.1,
    # Masterseed for NEST and NumPy.
    'master_seed': 46,
    # Number of threads per MPI process.
    'local_num_threads': 16,
    # If True, data will be overwritten,
    # If False, a NESTError is raised if the files already exist.
    'overwrite_files': True,
    # Print the time progress, this should only be used when the simulation
    # is run on a local machine.
    'print_time': True,
    # verbosity of Nest :M_ALL=0, M_DEBUG=5, M_STATUS=7, M_INFO=10, M_WARNING=20, M_ERROR=30, M_FATAL=40, M_QUIET=100
    'verbosity': 20
}

param_topology = {
    # Number of neurons
    'nb_neuron': 10 ** 4,
    # Ratio of inhibitory neurons
    'ratio_inhibitory': 0.2,
    # Parameter of excitatory neuron
    'excitatory_param':{
        'C_m': 200.0,
        't_ref': 5.0,
        'V_reset': -55.0,
        'E_L': -63.0,
        'g_L': 10.0,
        'I_e': 0.0,
        'a': 0.0,
        'b': 0.0,
        'Delta_T': 2.0,
        'tau_w': 500.0,
        'V_th': -50.0,
        'E_ex': 0.0,
        'tau_syn_ex': 5.0,
        'E_in': -80.0,
        'tau_syn_in': 5.0,
        'V_peak': 0.0,
    },
    # Parameter of inhibitory neuron
    'inhibitory_param':{
        'C_m': 200.0,
        't_ref': 5.0,
        'V_reset': -65.0,
        'E_L': -65.,
        'g_L': 10.0,
        'I_e': 0.0,
        'a': 0.0,
        'b': 0.0,
        'Delta_T': 0.5,
        'tau_w': 1.0,
        'V_th': -50.0,
        'E_ex': 0.0,
        'tau_syn_ex': 5.0,
        'E_in': -80.0,
        'tau_syn_in': 5.0,
        'V_peak': 0.0,
    },
    # Standard deviation of initial condition
    'sigma_V_0': 100.0,
    # Mean deviation of initial condition
    'mean_w_0': 200.0,
}

param_connexion = {
    # probability of excitatory connections
    'p_connect_ex': 0.05,
    # probability of inhibitory connections
    'p_connect_in': 0.05,
    # weight from excitatory neurons
    'Q_e': 1.5,
    # weight from inhibitory neurons
    'Q_i': 5.0
}

param_background = {
    # the mean firing rate of sinusoidal poisson_generator
    'rate': 0.0,
    # the weight on the connexion
    'weight': param_connexion['Q_e'],
}
