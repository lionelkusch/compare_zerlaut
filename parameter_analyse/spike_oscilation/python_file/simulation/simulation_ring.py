import nest
import numpy as np
import os
import subprocess
import time
import sys


def network_initialisation(results_path, param_nest):
    """
    Initialise the kernel of Nest
    :param results_path: Folder for saving the result of device
    :param param_nest: Dictionary with the parameter for Nest
    :return: Random generator for each thread
    """
    master_seed = param_nest['master_seed']
    local_num_threads = param_nest['local_num_threads']
    # Numpy random generator
    np.random.seed(master_seed)
    pyrngs = [np.random.RandomState(s) for s in range(master_seed, master_seed + local_num_threads)]
    # Nest Kernel
    nest.set_verbosity(param_nest['verbosity'])
    nest.ResetKernel()
    nest.SetKernelStatus({
        # Resolution of the simulation (in ms).
        "resolution": param_nest['sim_resolution'],
        # Print the time progress, this should only be used when the simulation
        # is run on a local machine.
        "print_time": True,
        # If True, data will be overwritten,
        # If False, a NESTError is raised if the files already exist.
        "overwrite_files": True,
        # Number of threads per MPI process.
        'local_num_threads': local_num_threads,
        # Path to save the output data
        'data_path': results_path,
        # Masterseed for NEST and NumPy
        'grng_seed': master_seed + local_num_threads,
        # Seeds for the individual processes
        'rng_seeds': range(master_seed + 1 + local_num_threads, master_seed + 1 + (2 * local_num_threads)),
    })
    return pyrngs


def network_initialisation_neurons(results_path, pyrngs, param_topology):
    """
    Create all neuron in each unit in Nest. An unit is composed by two populations (excitatory and inhibitory)
    :param results_path: Folder for saving the result of device
    :param pyrngs: Random generator for each thread
    :param param_topology: Dictionary with the parameter for the topology
    :return: Dictionary with the id of different layer and the index of the layer out
    """
    model = 'aeif_cond_exp'
    n_inh = int(param_topology['nb_neuron']*param_topology['ratio_inhibitory'])
    n_ex = int(param_topology['nb_neuron']*(1-param_topology['ratio_inhibitory']))
    sigma_init_V0 = param_topology['sigma_V_0']
    mean_init_w0 = param_topology['mean_w_0']

    # set the excitatory neurons
    pop_ex = nest.Create(model, n_ex)
    nest.SetStatus(pop_ex, param_topology['excitatory_param'])

    # set the inhibitory neurons
    pop_inh = nest.Create(model, n_inh)
    nest.SetStatus(pop_inh, param_topology['inhibitory_param'])

    for thread in np.arange(nest.GetKernelStatus('local_num_threads')):
        # Get all nodes on the local thread
        # Using GetNodes is a work-around until NEST 3.0 is released. It
        # will issue a deprecation warning.
        local_nodes = nest.GetNodes({
            'model': model,
            'thread': thread
        }, local_only=True
                                    )[0]
        # Get number of current virtual process
        # vp is the same for all local nodes on the same thread
        vp = nest.GetStatus(local_nodes)[0]['vp']
        nest.SetStatus(
            local_nodes, 'V_m', pyrngs[vp].normal(
                nest.GetDefaults(model)['E_L'],
                sigma_init_V0,
                len(local_nodes)
            )
        )
        nest.SetStatus(
            local_nodes, 'w', pyrngs[vp].normal(
                mean_init_w0,
                mean_init_w0,
                len(local_nodes)
            )
        )
    # save id of each population
    pop_file = open(os.path.join(results_path, 'population_GIDs.dat'), 'w+')
    pop_file.write('%d  %d %s\n' % (pop_ex.tolist()[0], pop_ex.tolist()[-1], 'excitatory'))
    pop_file.write('%d  %d %s\n' % (pop_inh.tolist()[0], pop_inh.tolist()[-1], 'inhibitory'))

    return pop_ex, pop_inh


def network_connection(pop_ex, pop_inh, param_connexion):
    """
    Create the connexion between all the neurons
    :param pop_ex: excitatory neurons
    :param pop_inh: inhibitory neurons
    :param param_connexion: Parameter for the connexions
    :return: nothing
    """
    prbC_ex = param_connexion['p_connect_ex']
    prbC_in = param_connexion['p_connect_in']
    Qe = param_connexion['Q_e']
    Qi = param_connexion['Q_i']

    # excitatory connections
    conn_dict_ex = {'rule': 'pairwise_bernoulli', 'p': prbC_ex, 'allow_autapses': False, 'allow_multapses': False}
    nest.CopyModel("static_synapse", "excitatory", {"weight": Qe, "delay": nest.GetKernelStatus("min_delay")})
    nest.Connect(pop_ex, pop_ex, conn_dict_ex, syn_spec="excitatory")
    nest.Connect(pop_ex, pop_inh, conn_dict_ex, syn_spec="excitatory")
    # inhibitory connection
    conn_dict_inh = {'rule': 'pairwise_bernoulli', 'p': prbC_in, 'allow_autapses': False, 'allow_multapses': False}
    nest.CopyModel("static_synapse", "inhibitory", {"weight": -Qi, "delay": nest.GetKernelStatus("min_delay")})
    nest.Connect(pop_inh, pop_ex, conn_dict_inh, syn_spec="inhibitory")
    nest.Connect(pop_inh, pop_inh, conn_dict_inh, syn_spec="inhibitory")


def network_device(pop_ex, pop_inh,  min_time, time_simulation, param_background, param_topology, param_connexion):
    """
    Create and Connect different record or input device
    :param pop_ex: excitatory neurons
    :param pop_inh: inhibitory neurons
    :param min_time: Beginning time of recording
    :param time_simulation: End of simulation
    :return: the list of multimeter and spike detector
    """
    frequency = param_background['frequency']
    if param_background['rate_equals_amplitude']:
        rate = param_background['amplitude']
    else:
        rate = param_background['rate']
    amplitude = param_background['amplitude']
    Qe = param_background['weight']
    nb_excitatory = int(param_topology['nb_neuron']*(1-param_topology['ratio_inhibitory']))
    prbC_ex = param_connexion['p_connect_ex']
    # create input
    parrot = nest.Create('parrot_neuron', nb_excitatory)
    # create sinusoidal input
    Poisson_sin = nest.Create('sinusoidal_poisson_generator', nb_excitatory,
                              params={'rate': rate, 'frequency': frequency, 'amplitude': amplitude, 'phase': 0.0})
    nest.Connect(Poisson_sin, parrot, 'one_to_one', syn_spec="static_synapse")
    nest.CopyModel("static_synapse", "poisson", {"weight": Qe, "delay": nest.GetKernelStatus("min_delay")})
    conn_dict_ex = {'rule': 'pairwise_bernoulli', 'p': prbC_ex, 'allow_autapses': False, 'allow_multapses': False}
    nest.Connect(parrot, pop_inh, conn_dict_ex, syn_spec="poisson")
    nest.Connect(parrot, pop_ex, conn_dict_ex, syn_spec="poisson")

    # Spike Detector
    # parameter of spike detector
    param_spike_dec = {"start": min_time,
                       "stop": time_simulation,
                       'record_to': 'ascii',
                       'label': 'excitatory'
                       }
    nest.CopyModel('spike_recorder', 'spike_recorder_ex')
    nest.SetDefaults("spike_recorder_ex", param_spike_dec)
    M1_pop_ex = nest.Create('spike_recorder_ex')
    nest.Connect(pop_ex, M1_pop_ex)
    nest.CopyModel('spike_recorder_ex', 'spike_recorder_inh')
    nest.SetDefaults("spike_recorder_inh", {'label':'inhibitory'})
    M1_pop_inh = nest.Create('spike_recorder_inh')
    nest.Connect(pop_inh, M1_pop_inh)
    return M1_pop_ex.tolist()[0], M1_pop_inh.tolist()[0]



def simulate(results_path, begin, end,
             param_nest, param_topology, param_connexion, param_background):
    """
    Run one simulation of simple network
    :param results_path: the name of file for recording
    :param begin : time of beginning to record
    :param end : time of end simulation
    """
    # Initialisation of the network
    tic = time.time()
    pyrngs = network_initialisation(results_path, param_nest)
    excitaotry_neurons, inhibitory_neurons = network_initialisation_neurons(results_path, pyrngs, param_topology)
    toc = time.time() - tic
    print("Time to initialize the network: %.2f s" % toc)

    # Connection and Device
    tic = time.time()
    network_connection(excitaotry_neurons, inhibitory_neurons, param_connexion)
    id_spike_recorder_ex, id_spike_recorder_in = network_device(excitaotry_neurons, inhibitory_neurons, begin, end, param_background,
                                                               param_topology, param_connexion)
    toc = time.time() - tic
    print("Time to create the connections and devices: %.2f s" % toc)

    # Simulation
    tic = time.time()
    nest.Simulate(end)
    toc = time.time() - tic
    print("Time to simulate: %.2f s" % toc)


    # Concatenate the different spike files
    if subprocess.call([os.path.join(os.path.dirname(__file__), '../run/script.sh'), results_path, str(id_spike_recorder_ex), str(id_spike_recorder_in)]) == 1:
        sys.stderr.write('ERROR bad concatenation of spikes file\n')
        exit(1)
