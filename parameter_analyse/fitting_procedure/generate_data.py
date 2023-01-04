#  Copyright 2021 Aix-Marseille Université
# "Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements; and to You under the Apache License, Version 2.0. "
import numpy as np
import os
from .plot.helper_function import remove_outlier


def compute_rate(data, begin, end, nb):
    """
    Compute the firing rate
    :param data: the spike of all neurons between end and begin
    :param begin: the time of the first spike
    :param end: the time of the last spike
    :return: the mean and the standard deviation of firing rate, the maximum and minimum of firing rate
    """
    # get data
    n_fil = data[:]
    n_fil = n_fil.astype(int)
    # count the number of the same id
    count_of_n = np.bincount(n_fil)
    # compute the rate
    rate_each_n_incomplet = count_of_n / (end - begin)
    # fill the table with the neurons which are not firing
    rate_each_n = np.concatenate((rate_each_n_incomplet, np.zeros(-np.shape(rate_each_n_incomplet)[0] + nb + 1)))
    return rate_each_n[1:]


def generate_rates(parameters,
                   MAXfexc, MINfexc, nb_value_fexc,
                   MAXfinh, MINfinh, nb_value_finh,
                   MAXadaptation, MINadaptation, nb_value_adaptation,
                   MAXJump, MINJump,
                   nb_neurons, name_file, dt, tstop):
    """
    generate rate output firing rate of the neurons dependant of excitatory and inhibitory input firing rate
    :param parameters: parameters of neurons
    :param MAXfexc: maximum excitatory input firing rate
    :param MINfexc: minimum firing rate
    :param nb_value_fexc: number of different value of excitatory input firing rate
    :param MAXfinh: maximum inhibitory input firing rate
    :param MINfinh: minimum inhibitory input firing rate
    :param nb_value_finh: number of different value of inhibitory input firing rate
    :param MAXadaptation: maximum of adaptation
    :param MINadaptation: minimum of adaptation
    :param nb_value_adaptation: number of different value of adaptation
    :param MAXJump: maximum jump of output firing rate
    :param MINJump: minimum jump of output firing rate
    :param nb_neurons: number of trial ofr each condition
    :param name_file: name of file of saving
    :param dt: step of integration
    :param tstop: time of simulation
    :return:
    """
    import nest
    nest.set_verbosity(100)

    # create folder
    if not os.path.exists(name_file):
        os.makedirs(name_file)
    # initialisation of the parameter
    params = {'g_L': parameters['g_L'],
              'E_L': parameters['E_L'],
              'V_reset': parameters['V_reset'],
              'I_e': parameters['I_e'],
              'C_m': parameters['C_m'],
              'V_th': parameters['V_th'],
              't_ref': parameters['t_ref'],
              'tau_w': parameters['tau_w'],
              'Delta_T': parameters['Delta_T'],
              'b': parameters['b'],
              'a': parameters['a'],
              'V_peak': parameters['V_peak'],
              'E_ex': parameters['E_ex'],
              'E_in': parameters['E_in'],
              'tau_syn_ex': parameters['tau_syn_ex'],
              'tau_syn_in': parameters['tau_syn_in'],
              'gsl_error_tol': 1e-8
              }
    Number_connexion_ex = parameters['N_tot'] * parameters['p_connect_ex'] * (1 - parameters['g'])
    Number_connexion_in = parameters['N_tot'] * parameters['p_connect_in'] * parameters['g']
    simtime = tstop * 1e3
    dt = dt * 1e3
    master_seed = 0
    local_num_threads = 8

    # initialisation of variable
    fiSim = np.repeat(np.linspace(MINfinh, MAXfinh, nb_value_finh), nb_value_adaptation).reshape(
        nb_value_finh * nb_value_adaptation) * Number_connexion_in
    adaptation = np.repeat([np.linspace(MINadaptation, MAXadaptation, nb_value_adaptation)], nb_value_finh,
                           axis=0).reshape(
        nb_value_finh * nb_value_adaptation)
    feSim = np.zeros((nb_value_fexc, nb_value_finh * nb_value_adaptation))
    feOut = np.zeros((nb_value_fexc, nb_value_finh * nb_value_adaptation, nb_neurons))
    MAXdfex = (MAXfexc - MINfexc) / nb_value_fexc
    dFex = np.ones((nb_value_adaptation * nb_value_finh)) * MAXdfex
    index_end = np.zeros((nb_value_adaptation * nb_value_finh), dtype=np.int)
    index = np.where(index_end >= 0)
    while index[0].size > 0:
        step = np.min(index_end[index])
        index_min = index[0][np.argmin(index_end[index])]
        print(step, index_min, dFex[index_min], feOut[[step - 1], index_min, :], feOut[step, index_min, :],
              np.nanmean(remove_outlier(feOut[[step - 1], index_min])),
              np.nanmean(remove_outlier(feOut[[step], index_min])), feSim[step, index_min],
              fiSim[index_min], adaptation[index_min])

        # simulation
        simulation = False
        error = 1.0e-6

        while error > 1.0e-20 and not simulation:
            params['gsl_error_tol'] = error
            # initialisation of nest
            nest.ResetKernel()
            nest.SetKernelStatus({
                # Resolution of the simulation (in ms).
                "resolution": dt,
                # Print the time progress, this should only be used when the simulation
                # is run on a local machine.
                "print_time": True,
                # If True, data will be overwritten,
                # If False, a NESTError is raised if the files already exist.
                "overwrite_files": True,
                # Number of threads per MPI process.
                'local_num_threads': local_num_threads,
                # Path to save the output data
                'data_path': name_file,
                # Masterseed for NEST and NumPy
                'grng_seed': master_seed + local_num_threads,
                # Seeds for the individual processes
                'rng_seeds': range(master_seed + 1 + local_num_threads, master_seed + 1 + (2 * local_num_threads)),
            })

            # create the network
            nest.SetDefaults('aeif_cond_exp', params)
            neurons = nest.Create('aeif_cond_exp', index[0].size * nb_neurons)
            nest.SetStatus(neurons, "I_e", -np.repeat(adaptation[index], nb_neurons).ravel())
            poisson_generator_ex = nest.Create('poisson_generator', index[0].size * nb_neurons)
            poisson_generator_in = nest.Create('poisson_generator', index[0].size * nb_neurons)
            nest.SetStatus(poisson_generator_in, 'rate', np.repeat(fiSim[index], nb_neurons).ravel())
            nest.SetStatus(poisson_generator_ex, 'rate',
                           np.repeat(feSim[index_end[index].ravel(), index], nb_neurons) * Number_connexion_ex)
            nest.CopyModel("static_synapse", "excitatory",
                           {"weight": parameters['Q_e'], "delay": 1.0})
            nest.CopyModel("static_synapse", "inhibitory",
                           {"weight": -parameters['Q_i'], "delay": 1.0})
            nest.Connect(poisson_generator_ex, neurons, 'one_to_one', syn_spec="excitatory")
            nest.Connect(poisson_generator_in, neurons, 'one_to_one', syn_spec="inhibitory")

            # create spike detector
            spikes_dec = nest.Create("spike_recorder")
            nest.Connect(neurons, spikes_dec)
            try:
                nest.Simulate(simtime)
                simulation = True
            except nest.NESTError as exception:
                template = "An exception of type {0} occurred. Arguments:\n{1!r}"
                message = template.format(type(exception).__name__, exception.args)
                print(message)
                error = error / 10.0
        # compute firing rate
        data = nest.GetStatus(spikes_dec)[0]['events']['senders']
        feOut[index_end[index].ravel(), index] = compute_rate(data, 0.0, simtime, index[0].size * nb_neurons).reshape(
            index[0].size, nb_neurons) * 1e3
        jump = np.nanmean(remove_outlier(feOut[index_end.ravel(), np.arange(0, index_end.size)]), axis=1) \
               - np.nanmean(remove_outlier(feOut[(index_end - 1).ravel(), np.arange(0, index_end.size)]), axis=1)

        # rescale if jump to big
        update_index = np.where(np.logical_and(index_end >= 0, jump > MAXJump))
        feSim[index_end.ravel()[update_index], update_index] -= dFex[update_index]
        dFex[update_index] /= 2
        feSim[index_end.ravel()[update_index], update_index] += dFex[update_index]

        # increase external input if no spike (initial condition  of external input)
        update_index = np.where(np.logical_and(np.logical_and(index_end >= 0, jump <= MAXJump), jump < MINJump))
        feSim[index_end.ravel()[update_index], update_index] += dFex[update_index]
        dFex[update_index] += dFex[update_index] * 0.1

        # save the data and pass at next value
        update_index = np.where(np.logical_and(np.logical_and(index_end >= 0, jump <= MAXJump), jump >= MINJump))
        index_end[update_index] += 1
        index_end[np.where(index_end == nb_value_fexc)[0]] = -1

        update = np.where(np.logical_and(np.logical_and(index_end >= 0, jump <= MAXJump), jump > MINJump))
        feSim[index_end.ravel()[update], update] = feSim[index_end.ravel()[update] - 1, update] + dFex[update]
        update = np.where(np.logical_and(np.logical_and(index_end >= 0, jump <= MAXJump), dFex < MAXdfex))[0]
        dFex[update][np.where(dFex[update] < MAXdfex)] += dFex[update] * 0.1
        index = np.where(index_end >= 0)

    np.save(name_file + '/fout.npy', feOut)  # save output firing rate
    np.save(name_file + '/fin.npy', feSim)   # save input excitatory firing rate
