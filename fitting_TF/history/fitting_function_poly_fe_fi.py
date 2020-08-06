import numpy as np
from scipy.optimize import minimize
import sys
from nest_elephant_tvb.Tvb.modify_tvb.fitting_TF import error_relatif_mean,error_absolute_mean, \
    error_neg,index_max,index_error_relatif,index_error_absolute,error_relatif,error_absolute

# function of fitting
def poly_1(p, fe, fi, adaptation):
    fe =  fe      + p[32] * fi + p[33] *  fi ** 2 + p[34] * fi ** 3
    a   = p[0] + p[8]  * fi + p[16] * fi ** 2 + p[24] * fi ** 3
    b   = p[1] + p[9]  * fi + p[17] * fi ** 2 + p[25] * fi ** 3
    c_1 = p[2] + p[10] * fi + p[18] * fi ** 2 + p[26] * fi ** 3
    c_2 = p[3] + p[11] * fi + p[19] * fi ** 2 + p[27] * fi ** 3
    c_3 = p[4] + p[12] * fi + p[20] * fi ** 2 + p[28] * fi ** 3
    c_4 = p[5] + p[13] * fi + p[21] * fi ** 2 + p[29] * fi ** 3
    c_5 = p[6] + p[14] * fi + p[22] * fi ** 2 + p[30] * fi ** 3
    d   = p[7] + p[15] * fi + p[23] * fi ** 2 + p[31] * fi ** 3
    c = c_1 + c_2 * fe + c_3 * fe ** 2 + c_4 * fe ** 3 + c_5 * fe ** 4
    return a / (b + np.exp(-c)) + d

def poly_2(p, fe, fi, adaptation):
    fe =  fe      + p[40] * fi + p[41] *  fi ** 2 + p[42] * fi ** 3 + p[43] * fi ** 4
    a   = p[0] + p[8]  * fi + p[16] * fi ** 2 + p[24] * fi ** 3 + p[32] * fi ** 4
    b   = p[1] + p[9]  * fi + p[17] * fi ** 2 + p[25] * fi ** 3 + p[33] * fi ** 4
    c_1 = p[2] + p[10] * fi + p[18] * fi ** 2 + p[26] * fi ** 3 + p[34] * fi ** 4
    c_2 = p[3] + p[11] * fi + p[19] * fi ** 2 + p[27] * fi ** 3 + p[35] * fi ** 4
    c_3 = p[4] + p[12] * fi + p[20] * fi ** 2 + p[28] * fi ** 3 + p[36] * fi ** 4
    c_4 = p[5] + p[13] * fi + p[21] * fi ** 2 + p[29] * fi ** 3 + p[37] * fi ** 4
    c_5 = p[6] + p[14] * fi + p[22] * fi ** 2 + p[30] * fi ** 3 + p[38] * fi ** 4
    d   = p[7] + p[15] * fi + p[23] * fi ** 2 + p[31] * fi ** 3 + p[39] * fi ** 4
    c = c_1 + c_2 * fe + c_3 * fe ** 2 + c_4 * fe ** 3 + c_5 * fe ** 4
    return a / (b + np.exp(-c)) + d

def generate_res(poly_function,error_function,feSim,fiSim,adaptation,feOut):
    if poly_function == 1:
        polynomial = poly_1
    elif poly_function == 2:
        polynomial =poly_2
    else:
        raise Exception('bad choice for polynomial function')
    if error_function == 1:
        error = error_relatif
    elif error_function == 2:
        error = error_absolute
    else:
        raise Exception('bad choice for error function')
    def Res (p):
        res = polynomial(p,feSim,fiSim,adaptation)
        if np.sum(np.isnan(res)) != 0 :
            return 10000000
        a = np.where(res < 0 )[0].shape[0] * 1
        b = error(feOut,res)
        print("\r error :", a,b,end=" "); sys.stdout.flush()
        return b
    return Res,polynomial

def fitting_fe_fi(feOut,feSim, fiSim, adaptation,value_adp):

    for j in range(0,20):
        mask = np.where(adaptation == value_adp[j])[0]
        feOut_1 = feOut[mask]
        feSim_1 = feSim[mask]
        fiSim_1 = fiSim[mask]
        adaptation_1 = adaptation[mask]
        Res,polynomial = generate_res(2,0,feSim_1,fiSim_1,adaptation_1,feOut_1)

        for i in range(100):
            plsq = minimize(Res, P, method='nelder-mead',
                            options={
                                # 'adaptive':True,
                                'disp': True,
                                'maxiter': 50000})
            P = plsq.x
            print(P)
            fit = polynomial(P,feSim, fiSim, adaptation)
            print("\r negative ",error_neg(feOut,fit),
                  " max error ", index_max(feOut,fit,1)[0],
                  " error relative ", error_relatif_mean(feOut,fit) ,
                  " error relative max ", index_error_relatif(feOut,fit,1)[0] ,
                  " error absolute ", error_absolute_mean(feOut,fit),
                  " error relative max ", index_error_absolute(feOut,fit,1)[0] ,
                  "\n\n")
        fit = polynomial(P,feSim, fiSim, adaptation)
        error_max_val,index_max_val = index_max (feOut,fit,1)
        error_1_val,index_1 = index_error_relatif (feOut,fit,5)
        error_2_val,index_2 = index_error_absolute (feOut,fit,5)
        print("")
        print("step ", j, " negative ", error_neg(fit) ," max error ", index_max(feOut,fit,1)[0],
              " error relative ", error_relatif_mean(feOut,fit) , " error absolute ", error_absolute_mean(feOut,fit))
        print(" negative ", error_neg(fit))
        print(" frequency ex", feSim[index_max_val]*1e3)
        print(" frequency in", fiSim[index_max_val]*1e3)
        print(" adaptation", adaptation[index_max_val])
        print('error max ', error_max_val)
        print(" expected : ", feOut[index_max_val]*1e3)
        print(" got : ",fit[index_max_val] * 1e3)
        print(" frequency ex", feSim[index_1]*1e3)
        print(" frequency in", fiSim[index_1]*1e3)
        print(" adaptation", adaptation[index_1])
        print('error relatif ', error_1_val)
        print(" expected : ", feOut[index_1]*1e3)
        print(" got : ",fit[index_1] * 1e3)
        print(" frequency ex", feSim[index_2]*1e3)
        print(" frequency in", fiSim[index_2]*1e3)
        print(" adaptation", adaptation[index_2])
        print('error absolute ', error_2_val)
        print(" expected : ", feOut[index_2]*1e3)
        print(" got : ",fit[index_2] * 1e3)
        print(P)
    return P

# # Best result :
# P = [ 3.36745929e-05, -1.08376863e+00, -8.05376603e-02, -1.41321715e-01,
#       1.98485436e+02, -7.24657308e+04, -6.67386298e+06, -2.50884083e-01,
#       1.16588144e-01, -7.31116482e-01, -1.11044080e+00,  4.39673319e+00,
#       6.16521436e+03,  2.13249116e+07, -1.27938357e+09,  7.88661768e+00,
#       1.04853052e+01,  3.69683743e+01, -1.67349574e+01, -7.64980244e+02,
#      -4.23949943e+06,  1.65080084e+08,  3.51804062e+08, -1.41632520e+02,
#       1.10117822e+03, -8.03825151e+02, -6.15947179e+03,  2.92581308e+05,
#       1.48475124e+07, -3.13116384e+09,  9.00192369e+11,  1.96853635e+03,
#      -8.76923463e+03,  7.81292607e+04, -4.82967583e+04, -1.91661567e+06,
#       1.16186905e+08,  1.06632823e+10, -1.18004029e+13, -1.23427161e+04,
#      -2.59010011e-01,  3.97753858e+00, -2.27572382e+02,  2.47297082e+03]


# best : 0.00049 0.021 0.85
# P = [ 3.99921079e-05, -1.08377204e+00, -8.06075428e-02, -2.82932918e-01,
#       3.34242879e+02, -1.64558485e+04, -7.65012038e+07, -1.63996887e-01,
#       1.11633566e-01,  5.96535452e-01, -8.01556348e-02, -2.89459034e+01,
#       1.32240519e+04,  6.90068957e+07,  6.30673413e+08,  5.38582054e+00,
#       2.83613148e+01,  1.38170503e+02, -4.96421606e+01,  1.10924489e+03,
#       -1.75187408e+07, -7.65573226e+08,  1.05816136e+10, -9.17833017e+01,
#       -5.31455832e+02,  4.10113702e+03,  7.42450458e+02,  1.38238354e+06,
#       1.58394734e+08, -3.02661207e+09,  5.46980946e+10,  8.30519556e+02,
#       -1.91248813e-01, -1.08177015e+00,  1.71208332e+01]
