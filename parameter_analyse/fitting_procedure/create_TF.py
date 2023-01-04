#  Copyright 2021 Aix-Marseille Université
# "Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements; and to You under the Apache License, Version 2.0. "
from parameter_analyse.fitting_procedure.function_fitting_parameters import engin
from parameter_analyse.fitting_procedure.parameters import excitatory_param, inhibitory_param, params_all
import numpy as np

np.set_printoptions(precision=2)

# # initial simulation
# print("'P_e':", np.array2string(engin(
#     parameters=excitatory_param, parameters_all=params_all, excitatory=True,
#     MAXfexc=200., MINfexc=0., nb_value_fexc=500, MAXfout=200.0,
#     name_file_fig='/home/kusch/Documents/project/Zerlaut/compare_zerlaut/parameter_analyse/fitting_procedure/fitting/',
#     nb_value_adaptation=20,
#     print_error=False, print_details=False, print_details_raw=False),  precision=10, separator=', '), ",", sep='')
# print("'P_i':", np.array2string(engin(
#     parameters=inhibitory_param, parameters_all=params_all, excitatory=False,
#     MAXfexc=200., MINfexc=0., nb_value_fexc=500, MAXfout=200.0,
#     name_file_fig='/home/kusch/Documents/project/Zerlaut/compare_zerlaut/parameter_analyse/fitting_procedure/fitting/',
#     nb_value_adaptation=20,
#     print_error=False), precision=10, separator=', '), ",", sep='')

# fitting
print("'P_e':", np.array2string(engin(
    parameters=excitatory_param, parameters_all=params_all, excitatory=True,
    MAXfexc=50., MINfexc=0., nb_value_fexc=500, MAXfout=20.0,
    name_file_fig='/home/kusch/Documents/project/Zerlaut/compare_zerlaut/parameter_analyse/fitting_procedure/fitting_50hz/',
    nb_value_adaptation=20,
    print_error=True, print_details=False, print_details_raw=False),  precision=10, separator=', '), ",", sep='')
print("'P_i':", np.array2string(engin(
    parameters=inhibitory_param, parameters_all=params_all, excitatory=False,
    MAXfexc=50., MINfexc=0., nb_value_fexc=500, MAXfout=50.0,
    name_file_fig='/home/kusch/Documents/project/Zerlaut/compare_zerlaut/parameter_analyse/fitting_procedure/fitting_50hz/',
    nb_value_adaptation=20,
    print_error=True), precision=10, separator=', '), ",", sep='')

# # test_error
# print("'P_e':", np.array2string(engin(
#     parameters=excitatory_param, parameters_all=params_all, excitatory=True,
#     MAXfexc=200., MINfexc=0., nb_value_fexc=500, MAXfout=200.0,
#     name_file_fig='/home/kusch/Documents/project/Zerlaut/compare_zerlaut/parameter_analyse/fitting_procedure/fitting_50hz/',
#     nb_value_adaptation=20, fitting=False, print_error=True),  precision=10, separator=', '), ",", sep='')
# print("'P_i':", np.array2string(engin(
#     parameters=inhibitory_param, parameters_all=params_all, excitatory=False,
#     MAXfexc=200., MINfexc=0., nb_value_fexc=500, MAXfout=200.0,
#     name_file_fig='/home/kusch/Documents/project/Zerlaut/compare_zerlaut/parameter_analyse/fitting_procedure/fitting_50hz/',
#     nb_value_adaptation=20, fitting=False, print_error=True), precision=10, separator=', '), ",", sep='')
