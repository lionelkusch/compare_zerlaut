import numpy as np

# function error
def error_relatif( real, fit) :
    error = ((fit - real)/real)
    mask = np.where(real == 0)
    error [mask] = (fit[mask] - real[mask])*1e3
    return error
def error_absolute( real, fit) :
    return (real-fit)*1e3
def error_relatif_mean ( real, fit) :
    return np.mean(error_relatif(real,fit)**2)
def error_absolute_mean (real,fit) :
    return np.mean( error_absolute(real,fit) ** 2)
def error_neg (fit):
    return  np.where(fit < 0 )[0].shape
def error_max (real,fit):
    return np.abs(error_absolute(real, fit))
def error_max_rel (real,fit):
    error = np.abs(error_relatif(real, fit))
    return error
def index_max (real,fit,nb):
    errors = error_max(real,fit)
    index = np.argsort(errors)[-nb:]
    return [errors[index],index]
def index_max_rel (real,fit,nb):
    errors = error_max_rel(real,fit)
    index = np.argsort(errors)[-nb:]
    return [errors[index],index]
