import os
import matplotlib.pyplot as plt
from parameter_analyse.fitting_procedure.parameters import excitatory_param, inhibitory_param, params_all
from parameter_analyse.fitting_procedure.plot.helper_function import get_result_raw
from parameter_analyse.fitting_procedure.fitting_function_zerlaut import fitting_model_zerlaut


def get_result_fitting(path, parameters, parameters_all, excitatory,
                       MAXfexc=50., MINfexc=0., nb_value_fexc=500,
                       MAXfinh=40., MINfinh=0., nb_value_finh=20,
                       MAXadaptation=100., MINadaptation=0., nb_value_adaptation=20,
                       MAXfout=20., MINfout=0.0, MAXJump=1.0, MINJump=0.1,
                       nb_neurons=50):
    """
    get data generated
    :return:
    """
    name_file = path
    for name, value in parameters.items():
        name_file += name + '_' + str(value) + '/'
    result_n_brut, data = get_result_raw(name_file, MAXfout, MAXfexc, nb_value_fexc, nb_neurons,
                                         MINfinh, MAXfinh, nb_value_finh, MINadaptation, MAXadaptation,
                                         nb_value_adaptation, data_require=True)
    p_with, p_without, TF = fitting_model_zerlaut(data[:, 0], data[:, 1], data[:, 2], data[:, 3], parameters_all,
                                                  excitatory, print_result=False, save_result=name_file,
                                                  fitting=False)
    return result_n_brut, data, p_with, p_without, TF

## path of the data
path = os.path.dirname(__file__) + '/../..//fitting_procedure/fitting_50hz/'
excitatory_result = get_result_fitting(path, excitatory_param, params_all, True)
inhibitory_result = get_result_fitting(path, inhibitory_param, params_all, False)
## parameter of the figures
labelticks_size = 12
label_legend_size = 12
ticks_size = 12
linewidth = 1.0
linewidth_error = 0.2
alpha = 1.0

## make figure
fig, axs = plt.subplots(3, 2, figsize=(6.8, 9.0))
## no external current
plt.sca(axs[0, 0])
plt.title('I external = 0 pA', {"fontsize": labelticks_size})
for j in range(excitatory_result[0].shape[1]):
    plt.plot(excitatory_result[0][:, j, 0, 1] * 1e3,
             excitatory_result[4](excitatory_result[0][:, j, 0, 1],
                                  excitatory_result[0][:, j, 0, 2],
                                  excitatory_result[2],
                                  w=excitatory_result[0][:, j, 0, 3]) * 1e3, 'b', linewidth=linewidth)
for j in range(excitatory_result[0].shape[1]):
    plt.plot(excitatory_result[0][:, j, 0, 1] * 1e3,
             excitatory_result[4](excitatory_result[0][:, j, 0, 1],
                                  excitatory_result[0][:, j, 0, 2],
                                  excitatory_result[3],
                                  w=excitatory_result[0][:, j, 0, 3]) * 1e3, '--c', alpha=1.0, linewidth=linewidth)
plt.ylabel("mean field (Hz)", {"fontsize": labelticks_size}, labelpad=2.)
plt.yticks([0.0, 80.0, 160.0])
plt.ylim(ymax=160.0, ymin=0.0)
plt.xticks([0.0, 20.0, 40.0])
plt.xlim(xmax=40.0, xmin=0.0)
plt.tick_params(labelsize=ticks_size)
plt.annotate('A', xy=(-0.1, 0.90), xycoords='axes fraction', weight='bold', fontsize=labelticks_size)

## error with adaptation
plt.sca(axs[1, 0])
for k in range(excitatory_result[0].shape[0]):
    for j in range(excitatory_result[0].shape[1]):
        plt.plot([excitatory_result[0][k, j, 0, 1] * 1e3, excitatory_result[0][k, j, 0, 1] * 1e3],
                 [excitatory_result[0][k, j, 0, 0] * 1e3,
                  excitatory_result[4](excitatory_result[0][k, j, 0, 1],
                                  excitatory_result[0][k, j, 0, 2],
                                  excitatory_result[2],
                                  w=excitatory_result[0][k, j, 0, 3]) * 1e3
                  ], color='r', alpha=alpha, linewidth=linewidth_error)
plt.ylabel("error (Hz)", {"fontsize": labelticks_size}, labelpad=2.)
plt.yticks([0.0, 80.0, 160.0])
plt.ylim(ymax=160.0, ymin=0.0)
plt.xlabel("firing rate of excitatory input (Hz)", {"fontsize": labelticks_size}, labelpad=2.)
plt.xticks([0.0, 20.0, 40.0])
plt.xlim(xmax=40.0, xmin=0.0)
plt.tick_params(labelsize=ticks_size)
plt.annotate('C', xy=(-0.1, 0.90), xycoords='axes fraction', weight='bold', fontsize=labelticks_size)

## error without adaptation
plt.sca(axs[2, 0])
for k in range(excitatory_result[0].shape[0]):
    for j in range(excitatory_result[0].shape[1]):
        plt.plot([excitatory_result[0][k, j, 0, 1] * 1e3, excitatory_result[0][k, j, 0, 1] * 1e3],
                 [excitatory_result[0][k, j, 0, 0] * 1e3,
                  excitatory_result[4](excitatory_result[0][k, j, 0, 1],
                                  excitatory_result[0][k, j, 0, 2],
                                  excitatory_result[3],
                                  w=excitatory_result[0][k, j, 0, 3]) * 1e3
                  ], color='r', alpha=alpha, linewidth=linewidth_error)
plt.ylabel("error (Hz)", {"fontsize": labelticks_size}, labelpad=2.)
plt.yticks([0.0, 80.0, 160.0])
plt.ylim(ymax=160.0, ymin=0.0)
plt.xlabel("firing rate of excitatory input (Hz)", {"fontsize": labelticks_size}, labelpad=2.)
plt.xticks([0.0, 20.0, 40.0])
plt.xlim(xmax=40.0, xmin=0.0)
plt.tick_params(labelsize=ticks_size)
plt.annotate('E', xy=(-0.1, 0.90), xycoords='axes fraction', weight='bold', fontsize=labelticks_size)

## current maximal
plt.sca(axs[0, 1])
plt.title('I external = 100 pA', {"fontsize": labelticks_size})
for j in range(excitatory_result[0].shape[1]):
    plt.plot(excitatory_result[0][:, j, -1, 1] * 1e3,
             excitatory_result[4](excitatory_result[0][:, j, -1, 1],
                                  excitatory_result[0][:, j, -1, 2],
                                  excitatory_result[2],
                                  w=excitatory_result[0][:, j, -1, 3]) * 1e3, 'b', linewidth=linewidth)
for j in range(excitatory_result[0].shape[1]):
    plt.plot(excitatory_result[0][:, j, -1, 1] * 1e3,
             excitatory_result[4](excitatory_result[0][:, j, -1, 1],
                                  excitatory_result[0][:, j, -1, 2],
                                  excitatory_result[3],
                                  w=excitatory_result[0][:, j, -1, 3]) * 1e3, '--c', alpha=1.0, linewidth=linewidth)
plt.yticks([0.0, 80.0, 160.0])
plt.ylim(ymax=160.0, ymin=0.0)
plt.xticks([0.0, 20.0, 40.0])
plt.xlim(xmax=40.0, xmin=0.0)
plt.tick_params(labelsize=ticks_size)
plt.annotate('B', xy=(-0.1, 0.90), xycoords='axes fraction', weight='bold', fontsize=labelticks_size)

## error with adaptation
plt.sca(axs[1, 1])
for k in range(excitatory_result[0].shape[0]):
    for j in range(excitatory_result[0].shape[1]):
        plt.plot([excitatory_result[0][k, j, -1, 1] * 1e3, excitatory_result[0][k, j, -1, 1] * 1e3],
                 [excitatory_result[0][k, j, -1, 0] * 1e3,
                  excitatory_result[4](excitatory_result[0][k, j, -1, 1],
                                       excitatory_result[0][k, j, -1, 2],
                                       excitatory_result[2],
                                       w=excitatory_result[0][k, j, -1, 3]) * 1e3
                  ], color='r', alpha=alpha, linewidth=linewidth_error)
plt.yticks([0.0, 80.0, 160.0])
plt.ylim(ymax=160.0, ymin=0.0)
plt.xlabel("firing rate of excitatory input (Hz)", {"fontsize": labelticks_size}, labelpad=2.)
plt.xticks([0.0, 20.0, 40.0])
plt.xlim(xmax=40.0, xmin=0.0)
plt.tick_params(labelsize=ticks_size)
plt.annotate('D', xy=(-0.1, 0.90), xycoords='axes fraction', weight='bold', fontsize=labelticks_size)

## error without adaptation
plt.sca(axs[2, 1])
for k in range(excitatory_result[0].shape[0]):
    for j in range(excitatory_result[0].shape[1]):
        plt.plot([excitatory_result[0][k, j, -1, 1] * 1e3, excitatory_result[0][k, j, -1, 1] * 1e3],
                 [excitatory_result[0][k, j, -1, 0] * 1e3,
                  excitatory_result[4](excitatory_result[0][k, j, -1, 1],
                                       excitatory_result[0][k, j, -1, 2],
                                       excitatory_result[3],
                                       w=excitatory_result[0][k, j, -1, 3]) * 1e3
                  ], color='r', alpha=alpha, linewidth=linewidth_error)
plt.yticks([0.0, 80.0, 160.0])
plt.ylim(ymax=160.0, ymin=0.0)
plt.xlabel("firing rate of excitatory input (Hz)", {"fontsize": labelticks_size}, labelpad=2.)
plt.xticks([0.0, 20.0, 40.0])
plt.xlim(xmax=40.0, xmin=0.0)
plt.tick_params(labelsize=ticks_size)
plt.annotate('F', xy=(-0.1, 0.90), xycoords='axes fraction', weight='bold', fontsize=labelticks_size)

plt.subplots_adjust(top=0.975, bottom=0.06, left=0.100, right=0.985, wspace=0.17, hspace=0.12)
plt.savefig('./figure/SP_figure_7.png', dpi=300)
# plt.show()