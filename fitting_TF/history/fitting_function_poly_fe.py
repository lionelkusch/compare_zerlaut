import numpy as np
import pathos.multiprocessing as mp
import dill
from scipy.optimize import minimize
from itertools import product
import time
from nest_elephant_tvb.Tvb.modify_tvb.fitting_TF import error_relatif_mean,error_absolute_mean,\
    error_neg,index_max,index_error_relatif,index_error_absolute


# polynome test for only excitatory firing rate
# first degree
def f_1(p, fe, fi, adaptation):
    a = p[0]
    b = p[1]
    c = p[2] + p[4] * fe
    d = p[3] + p[5] * fe
    return a / (b + np.exp(-c)) + d

def f_2(p, fe, fi, adaptation):
    a = p[0]
    b = p[1]
    c = p[2] + p[4] * fe + p[6] * fe ** 2
    d = p[3] + p[5] * fe + p[7] * fe ** 2
    return a / (b + np.exp(-c)) + d

def f_3(p, fe, fi, adaptation):
    a = p[0]
    b = p[1]
    c = p[2] + p[4] * fe + p[6] * fe ** 2 + p[8] * fe ** 3
    d = p[3] + p[5] * fe + p[7] * fe ** 2 + p[9] * fe ** 3
    return a / (b + np.exp(-c)) + d

def f_4(p, fe, fi, adaptation):
    a = p[0]
    b = p[1]
    c = p[2] + p[4] * fe + p[5] * fe ** 2 + p[6] * fe ** 3
    d = p[3]
    return a / (b + np.exp(-c)) + d

# Best solution : firing rate estimation under 1Hz
def f_5(p, fe, fi, adaptation):
    a = p[0]
    b = p[1]
    c = p[2] + p[4] * fe + p[5] * fe ** 2 + p[6] * fe ** 3 + p[7] * fe ** 4
    d = p[3]
    return a / (b + np.exp(-c)) + d

def fitting_fe(result,save_file):
    """
    fitting  of polynomial only for fixed adaptation and inhibitory firing rate
    :param result:
    :return:
    """
    def Res(f,feOut,feSim,fiSim,adaptation):
        def error_func(p):
            res = f(p, feSim, fiSim, adaptation)
            if np.sum(np.isnan(res)) != 0:
                return 10000000
            a = np.where(res < 0)[0].shape[0] * 1
            mask = np.where(res >= 0)
            b = error_relatif_mean(feOut[mask],res[mask])
            return a + b
        return error_func
    function = f_5 # choice of polynomial function
    save_poly = [] # save all polynomeial
    p = mp.ProcessingPool(nodes=16) # for multi processing
    explore = list(product(range(result.shape[2]),range(result.shape[1]))) # adaptation, inhbition
    def fitting(a):
        # get data
        inh = a[1]
        adp = a[0]
        feOut = result[:,inh,adp,0]
        feSim = result[:,inh,adp,1]
        fiSim = result[:,inh,adp,2]
        adaptation = result[:,inh,adp,3]
        mask = np.logical_not(np.isnan(feOut) +np.isnan(feSim) + np.isnan(fiSim) +np.isnan(adaptation))
        feOut = feOut[mask]
        feSim = feSim[mask]
        fiSim = fiSim[mask]
        adaptation = adaptation[mask]
        print(" index inh : ", inh, " index adaptation ", adp )

        # different initial condition of the function
        def init(P):
            tic = time.time()
            for i in range(100):
                if time.time() - tic > 300:
                    break
                plsq = minimize(Res(function, feOut, feSim, fiSim, adaptation),
                                P, method='Nelder-Mead', options={
                        'disp': False,
                        'maxiter': 50000})
                P = plsq.x
            fit = function(P, feSim, fiSim, adaptation)
            return (error_relatif_mean(feOut, fit), P)

        results =[]
        results.append(init(np.zeros(8))) # start with the function zeros
        results.append(init(np.ones(8))) # start with a function complexe
        results.append(init([0.01, 1.0, -10., 500., 0., 0., 0., 0.])) # start with somme good approximation
        results.append(init( [0.1, 1.0, -10., 500., 0., 0., 0., 0.])) # start with a different scaling
        error_good = None
        poly_good = []
        i = 0
        for error, poly in results:
            fit = function(poly,feSim, fiSim, adaptation)
            print("step  : ",i, " ", adp, " ", inh," negative ", error_neg(fit) ," max error ", index_max(feOut,fit,1)[0],
                  " error relative ", error_relatif_mean(feOut,fit) , " error absolute ", error_absolute_mean(feOut,fit))
            i+=1
            if error_good is None or error < error_good:
                error_good = error
                poly_good = poly

        fit = function(poly_good,feSim, fiSim, adaptation)
        error_max_val,index_max_val = index_max (feOut,fit,1)
        error_1_val,index_1 = index_error_relatif (feOut,fit,1)
        error_2_val,index_2 = index_error_absolute (feOut,fit,1)
        print("")
        print("step ", adp, " ", inh," negative ", error_neg(fit) ," max error ", index_max(feOut,fit,1)[0],
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
        print('error 1 ', error_1_val)
        print(" expected : ", feOut[index_1]*1e3)
        print(" got : ",fit[index_1] * 1e3)
        print(" frequency ex", feSim[index_2]*1e3)
        print(" frequency in", fiSim[index_2]*1e3)
        print(" adaptation", adaptation[index_2])
        print('error 2 ', error_2_val)
        print(" expected : ", feOut[index_2]*1e3)
        print(" got : ",fit[index_2] * 1e3)
        print(poly_good)
        return [error_max_val,error_1_val,error_1_val,poly_good]
    save_poly = p.map(dill.copy(fitting),explore)
    np.save(save_file+"poly_2_fe_ex.npy", np.array(save_poly))
    error_max_val = 1000.0
    for i in save_poly:
        if i[0] > error_max_val:
            error_max_val = i[0]
    print(" error max of approximation: ",error_max_val)
    p.terminate()
