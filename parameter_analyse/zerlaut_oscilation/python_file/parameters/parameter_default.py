import numpy as np
import os


class Parameter:
    def __init__(self):
        path = os.path.dirname(os.path.abspath(__file__))
        self.parameter_simulation = {
            'path_result': '/home/kusch/Documents/project/Zerlaut/compare_zerlaut/parameter_analyse/zerlaut_oscilation/simulation/',
            'seed': 10,  # the seed for the random generator
            'save_time': 1000.0,  # the time of simulation in each file
        }

        self.parameter_model = {
            # order of the model
            'order': 2,
            # parameter of the model
            'g_L': 10.0,
            'E_L_e': -63.0,
            'E_L_i': -65.0,
            'C_m': 200.0,
            'b_e': 0.0,
            'a_e': 0.0,
            'b_i': 0.0,
            'a_i': 0.0,
            'tau_w_e': 500.0,
            'tau_w_i': 1.0,
            'E_e': 0.0,
            'E_i': -80.0,
            'Q_e': 1.5,
            'Q_i': 5.0,
            'tau_e': 5.0,
            'tau_i': 5.0,
            'N_tot': 10000,
            'p_connect_e': 0.05,
            'p_connect_i': 0.05,
            'g': 0.2,
            'T': 5.0,
            'P_e': [-0.0489345812, -0.00219309  ,  0.0478268882, -0.0200377004,  0.0012958861, 0.0192032604,  0.0935502423, -0.0105511159,  0.0025571267,  0.0639695697],
            'P_i': [-0.0505735018,  0.0006669982,  0.0206049386, -0.0065279486,  0.0007504186, 0.016254991 ,  0.0258427852, -0.0073418624,  0.0030557889,  0.0109887961],
            'K_ext_e': 400,  # int(parameter_model['N_tot'] * parameter_model['p_connect_e'] * (1 - parameter_model['g']))
            'K_ext_i': 0,
            # noise parameters
            'tau_OU': np.inf,
            'weight_noise': 0.0,
            'S_i': 0.0,
            # Initial condition :
            'initial_condition': {
                "E": [0.000, 0.000], "I": [0.00, 0.00], "C_ee": [0.0, 0.0], "C_ei": [0.0, 0.0], "C_ii": [0.0, 0.0],
                "W_e": [100.0, 100.0], "W_i": [0.0, 0.0], "noise": [0.0, 0.0],
                "external_input_excitatory_to_excitatory": [0.0, 0.0],
                "external_input_excitatory_to_inhibitory": [0.0, 0.0],
                "external_input_inhibitory_to_excitatory": [0.0, 0.0],
                "external_input_inhibitory_to_inhibitory": [0.0, 0.0],
            }
        }

        self.parameter_connection_between_region = {
            ## CONNECTIVITY
            # File description
            'number_of_regions': 50,  # number of region
        }

        self.parameter_coupling = {
            ##COUPLING
            'type': 'Linear',
            # choice : Linear, Scaling, HyperbolicTangent, Sigmoidal, SigmoidalJansenRit, PreSigmoidal, Difference, Kuramoto
            'parameter': {'a': 0.0,
                          'b': 0.0}
        }

        self.parameter_integrator = {
            ## INTEGRATOR
            'type': 'Heun',  # choice : Heun, Euler
            'stochastic': True,
            'noise_type': 'Additive',  # choice : Additive
            'noise_parameter': {
                'nsig': [1.0*1e-6, 1.0*1e-6, 0.0, 0.0, 0.0, 0.0, 0.0],
                'ntau': 0.0,
                'dt': 0.1
            },
            'dt': 0.1  # in ms
        }

        self.parameter_monitor = {
            'Raw': True,
            'TemporalAverage': False,
            'parameter_TemporalAverage': {
                'variables_of_interest': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                'period': self.parameter_integrator['dt'] * 10.0
            },
            'Bold': False,
            'parameter_Bold': {
                'variables_of_interest': [0],
                'period': self.parameter_integrator['dt'] * 2000.0
            }
            # More monitor can be added
        }

        self.parameter_stimulus = {
            'amp': (np.arange(1., 51., 1.)*1e-3).tolist(),
            "frequency": 1.0*1e-3,
            "weights": np.ones((self.parameter_connection_between_region['number_of_regions'], 1)).tolist(),
            "variables": [8, 9]
        }
