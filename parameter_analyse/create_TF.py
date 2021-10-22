from parameter_analyse.function_fitting_parameters import engin
from parameters import excitatory_param, inhibitory_param, params_all
import numpy as np

np.set_printoptions(precision=2)

print("'P_e':", np.array2string(engin(parameters=excitatory_param, parameters_all=params_all, excitatory=True,
                                      MAXfexc=200., MINfexc=0., nb_value_fexc=500,
                                      MAXfout=200.0,
                                      name_file_fig='/home/kusch/Documents/project/Zerlaut/compare_zerlaut/parameter_analyse/fitting/',
                                      nb_value_adaptation=20,
                                      print_error=True),  precision=10, separator=', '), ",", sep='')
print("'P_i':", np.array2string(engin(parameters=inhibitory_param, parameters_all=params_all, excitatory=False,
                                      MAXfexc=200., MINfexc=0., nb_value_fexc=500,
                                      MAXfout=200.0,
                                      name_file_fig='/home/kusch/Documents/project/Zerlaut/compare_zerlaut/parameter_analyse/fitting/',
                                      nb_value_adaptation=20,
                                      print_error=True), precision=10, separator=', '), ",", sep='')

# more data 40hz -200hz
# P_e = [-4.9229490413e-02,  1.7647136306e-03, -8.0191842361e-04, -3.7988001243e-03,  2.3460007281e-04,  4.0164132129e-03, 1.7666175010e-03, -2.1593867802e-05,  1.8048983267e-04, 3.9327144896e-03]
# P_i = [-0.0507994447,  0.0018667585, -0.000361398 , -0.0003923391,  0.0005466599, 0.0017309422, -0.0023381279,  0.0009409149,  0.0014449343, -0.0010911468]
# 40hz - 40hz
# P_e = [-0.0492369084,  0.0017828621, -0.0009860554, -0.0037848567,  0.0002349566,  0.0039129801,  0.0015830248,  0.0001158348,  0.0002085719,  0.003820842 ]
# P_i = [-0.0507843127,  0.0018199046,  0.000165465 , -0.0006345173,  0.0005507706,  0.0018760524, -0.0015854286,  0.0007536946,  0.0015902651, -0.0004062734]