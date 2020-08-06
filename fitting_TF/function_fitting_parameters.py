import logging
logging.getLogger('numba').setLevel('ERROR')
logging.getLogger('matplotlib').setLevel('ERROR')
logging.getLogger('tvb').setLevel('ERROR')
import os
import numpy as np
from nest_elephant_tvb.Tvb.modify_tvb.fitting_TF.generate_data import generate_rates,remove_outlier
from nest_elephant_tvb.Tvb.modify_tvb.fitting_TF.New_parameters import excitatory_param,inhibitory_param
from nest_elephant_tvb.Tvb.modify_tvb.fitting_TF.fitting_function_poly import fitting_all,poly
from nest_elephant_tvb.Tvb.modify_tvb.fitting_TF.fitting_function_zerlaut import fitting_1
from nest_elephant_tvb.Tvb.modify_tvb.fitting_TF.print_fitting_figure import print_result_box_plot,print_result_std,print_result_curve_box_std,print_result,print_result_1,print_result_zerlaut

def engin(parameters,excitatory,
          MAXfexc=40., MINfexc=0., nb_value_fexc=60,
          MAXfinh=40., MINfinh=0., nb_value_finh=20,
          MAXadaptation=100.,MINadaptation=0., nb_value_adaptation=20,
          MAXfout=20., MINfout=0.0, MAXJump=1.0, MINJump=0.1,
          nb_neurons=50,
          name_file_fig='/home/kusch/Documents/project/co_simulation/co-simulation-tvb-nest/example/fitting/',
          dt=1e-4, tstop=10.0):
    name_file = name_file_fig
    for name,value in parameters.items():
        name_file += name+'_'+str(value)+'/'

    if os.path.exists(name_file+'/P.npy'):
        return np.load(name_file+'/P.npy')
    elif os.path.exists(name_file + '/fout.npy'):
        pass
    else:
         generate_rates(parameters,
                        MAXfexc=MAXfexc, MINfexc=MINfexc, nb_value_fexc=nb_value_fexc,
                        MAXfinh=MAXfinh, MINfinh=MINfinh, nb_value_finh=nb_value_finh,
                        MAXadaptation=MAXadaptation,MINadaptation=MINadaptation, nb_value_adaptation=nb_value_adaptation,
                        MAXJump=MAXJump, MINJump=MINJump,
                        nb_neurons=nb_neurons, name_file=name_file,dt=dt, tstop=tstop
                        )
    # structure data
    results = np.load(name_file + '/fout.npy').reshape(nb_value_adaptation*nb_value_fexc*nb_value_finh,nb_neurons)*1e-3
    feOut = np.nanmean(remove_outlier(results), axis= 1)
    feOut_std = np.nanstd(remove_outlier(results), axis = 1 ).ravel()
    feOut_med = np.median(remove_outlier(results), axis = 1 ).ravel()
    feSim = np.load(name_file + '/fin.npy').ravel() *1e-3
    fiSim = np.repeat([np.repeat(np.linspace(MINfinh, MAXfinh, nb_value_finh), nb_value_adaptation)], nb_value_fexc,
                      axis=0).ravel() *1e-3
    adaptation = np.repeat(
        [np.repeat([np.linspace(MINadaptation, MAXadaptation, nb_value_adaptation)], nb_value_finh, axis=0)],
        nb_value_fexc, axis=0).ravel()

    i=0
    result_n = np.empty((nb_value_fexc,nb_value_finh,nb_value_adaptation,6+nb_neurons))
    result_n[:]=np.NAN
    fe_model = -1
    np.linspace(MINadaptation, MAXadaptation, nb_value_adaptation)
    data = []
    while i != len(fiSim):
        fi_model = np.where(fiSim[i]==np.linspace(MINfinh,MAXfinh, nb_value_finh)*1e-3)[0][0]
        w_model = np.where(adaptation[i]==np.linspace(MINadaptation, MAXadaptation, nb_value_adaptation))[0][0]
        if adaptation[i] < adaptation[i-1]:
            if fiSim[i] < fiSim[i-1]:
                fe_model += 1
        result_n[fe_model, fi_model, w_model, :6] = [feOut[i], feSim[i], fiSim[i], adaptation[i], feOut_std[i],feOut_med[i]]
        result_n[fe_model, fi_model, w_model, 6:] = results[i]
        if not(feOut[i] > MAXfout*1e-3): # coefficient of variation small and variation under 2 Hz
            data.append([feOut[i], feSim[i], fiSim[i], adaptation[i]])
        i+=1
    data = np.array(data)
    # Print data
    if excitatory:
        name_file_fig+="/ex_"
    else:
        name_file_fig+="/in_"
    # print_result_box_plot(result_n,name_file_fig,nb_value_finh,nb_value_fexc)
    # print_result_std(result_n,name_file_fig)
    # print_result_curve_box_std(result_n,name_file_fig,nb_value_finh,nb_value_fexc)

    # Fitting data Zerlaut
    p_with,p_without,TF = fitting_1(data[:,0],data[:,1],data[:,2],data[:,3], parameters, nb_value_fexc,nb_value_finh,nb_value_adaptation,
              MINadaptation, MAXadaptation,MINfinh,MAXfinh,MAXfexc,excitatory)
    print_result_zerlaut(result_n,TF,p_with,p_without,name_file_fig+'zerlaut_error',nb_value_finh,nb_value_fexc)
    # Fitting polynomial
    # p_with = fitting_all(data[:,0],data[:,1],data[:,2],data[:,3],max_iter=10)
    # p_with = fitting_all(result_n[:,:,:,5].ravel(),result_n[:,:,:,1].ravel(),result_n[:,:,:,2].ravel(),result_n[:,:,:,3].ravel())

    # Excitatory
    if excitatory:
        P_relatif = [-9.00589681e-02, -7.42395590e-01,  1.64790007e-01,  7.17642501e+00,
                     -7.91176598e+03, -9.50354089e+05,  5.81795634e+07,  8.62688969e-01,
                     -6.04072032e-05,  4.10899185e-05, -2.71944385e-05, -1.89300627e-02,
                      6.92868021e+00, -1.09126882e+03,  6.81784465e+04,  7.94495542e-06,
                     -7.11213528e+01,  1.11917608e+02,  3.37302892e+01, -7.28063752e+01,
                      1.23371481e+06, -1.37638536e+08,  3.45337328e+09, -5.98567013e+00,
                     -5.21987178e-06, -1.40919625e-01,  1.76881880e+02, -1.21617315e+02,
                      1.37630938e+01, -6.96904647e+04, -2.22326548e+07,  5.21731096e+09,
                     -1.78926671e+11,  2.13108605e+01, -5.38516713e+00,  3.89118632e+03,
                      6.20446048e+01, -1.43178776e+03,  8.69668203e+05,  1.27187045e+08,
                     -5.37716294e+10,  8.48361913e+11, -2.79668349e+02,  9.90181221e+01,
                      5.55036990e+03,  8.05614615e+03,  7.46356424e+04, -2.33797407e+06,
                      1.33063204e+09, -1.67718975e+11,  2.86919537e+13, -2.43577230e+03,
                     -1.04488962e+03]
        # p_with = fitting_all(data[:,0],data[:,1],data[:,2],data[:,3],max_iter=20,P=P_relatif)

        P_absolute = [-1.28215423e-01, -7.11957088e-01,  1.51404311e-01,  3.55159410e+00,
                     -7.74029279e+03, -6.41886122e+05,  3.57275446e+08,  8.67200762e-01,
                     -2.40284496e-04,  2.63562089e-04, -1.59201328e-05,  8.46801016e-03,
                     -6.07099536e+00,  3.14539971e+03, -2.79297988e+05,  1.89575403e-05,
                     -7.24225665e+01,  1.12331876e+02,  3.20530472e+01,  2.14811881e+01,
                      4.30407779e+05, -2.03353217e+08,  1.29783583e+09, -6.30070165e+00,
                     -4.30720109e-06, -2.19330079e-01,  3.03166300e+02, -1.20226242e+02,
                      9.54663509e+01, -2.43631268e+04, -3.86287157e+06,  6.88880593e+09,
                     -4.22521197e+11,  2.46750156e+01, -8.45025447e+00,  4.67649563e+03,
                     -2.65451498e+03, -6.40776167e+02, -1.44923200e+05, -3.11556743e+08,
                     -9.13465816e+10,  1.03568408e+13, -7.57000933e+02,  1.17392520e+02,
                      2.08540571e+04,  1.92957818e+02,  4.47174478e+04,  1.61177285e+07,
                      5.67229902e+09,  4.45074504e+11, -9.35400810e+13,  6.10711521e+01,
                      0.0]
    else:
        P_relatif = [ 4.62640386e-03, -1.60263399e+00, -4.82618284e-01, -2.85592315e+00,
                      3.55735392e+03,  4.36121194e+05,  5.77631251e+06, -2.49183323e-01,
                      1.94023554e-05,  1.07459955e-04,  1.60297243e-05,  6.99812250e-03,
                      -3.58760652e+00,  5.60439874e+02, -3.45074629e+04,  4.54339345e-06,
                      4.75851959e+00,  4.32939661e+01,  1.52486219e+01, -3.69241724e+02,
                      -5.24210904e+05,  8.04236488e+07, -2.99721794e+09, -2.15972585e+00,
                      -4.32915579e-06, -5.20013063e-02,  5.06536955e+02,  1.17886480e+03,
                      -3.10554781e+02,  7.47938979e+04, -1.55630405e+06, -2.22217018e+09,
                      9.51638164e+10,  1.38856189e+01, -9.10006451e+00, -1.20851322e+03,
                      -2.07354767e+03,  3.25514222e+03, -9.81067771e+05,  9.69584325e+07,
                      1.04980999e+10, -6.40090805e+10, -8.72269744e+01,  2.44727515e+02,
                      4.53308904e+03, -1.90861273e+04,  1.24028643e+04,  6.15360284e+06,
                      1.80661772e+09, -1.71477940e+11, -2.63231820e+12, -3.86034305e+03,
                      -2.30302648e+03]
        # p_with = fitting_all(data[:,0],data[:,1],data[:,2],data[:,3],max_iter=20,P=P_relatif)


        P_absolute = [ 1.05412523e-02, -1.59611042e+00, -4.98800039e-01, -2.73687732e+00,
                      7.01100139e+03,  3.56826299e+05, -3.40024437e+07, -2.09649575e-01,
                      2.70407383e-05,  2.28733821e-04,  5.94955901e-05,  7.50518888e-03,
                     -1.45704138e+00,  6.20314013e+01,  1.41532697e+03,  7.31582451e-06,
                      6.92673971e+00,  5.58565104e+01,  1.46945925e+01, -1.33658276e+03,
                     -4.71835951e+05,  8.30549556e+07, -2.06748130e+09, -3.12083983e+00,
                     -4.57027151e-06, -7.71722577e-02,  5.08945478e+02,  8.07373552e+02,
                     -5.28836355e+02,  9.07176722e+04, -1.30340316e+06, -2.23370768e+09,
                      8.83413554e+10, -9.61792134e+00, -7.66185229e+00, -1.64996341e+03,
                     -7.39227956e+03,  4.53448698e+03, -1.11452566e+06,  1.39150623e+08,
                      1.36436661e+10, -2.80527759e+11, -1.16118770e+02,  8.44679046e+01,
                      3.59260506e+04, -4.69827051e+04,  5.11620195e+04, -3.24114309e+06,
                      4.31942611e+08, -1.24180209e+11, -4.24740632e+12, -1.63277065e+04,
                      0.0]

    # print_result(result_n,poly(7),P_relatif,P_absolute,name_file_fig+'poly_2_',nb_value_finh,nb_value_fexc)
    print_result_1(result_n,poly(7),P_absolute,name_file_fig+'poly_2_abs_',nb_value_finh,nb_value_fexc)
    print_result_1(result_n,poly(7),P_relatif,name_file_fig+'poly_2_rel_',nb_value_finh,nb_value_fexc)



    return p_with


# print("'P_e':",np.array2string(engin(parameters=excitatory_param,excitatory=True), separator=', '),",",sep='')
print("'P_i':",np.array2string(engin(parameters=inhibitory_param,excitatory=False), separator=', '),",",sep='')


# print("'P_e':",np.array2string(engin(parameters=excitatory_param,excitatory=True,
#                                      name_file_fig='/home/kusch/Documents/project/co_simulation/co-simulation-tvb-nest/example/fitting_test/',
#                                      nb_value_adaptation=5), separator=', '),",",sep='')
# print("'P_i':",np.array2string(engin(parameters=inhibitory_param,excitatory=False,
#                                      name_file_fig='/home/kusch/Documents/project/co_simulation/co-simulation-tvb-nest/example/fitting_test/',
#                                      nb_value_adaptation=5), separator=', '),",",sep='')
