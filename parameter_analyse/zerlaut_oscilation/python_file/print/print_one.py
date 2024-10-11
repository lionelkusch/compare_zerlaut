#  Copyright 2023 Aix-Marseille Université
# "Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements; and to You under the Apache License, Version 2.0. "
import parameter_analyse.zerlaut_oscilation.python_file.run.tools_simulation as tools
import matplotlib.pyplot as plt


def plot_result(path_simulation, begin, end, region = 19):
    """
    plot result
    :param path_simulation: path of the simulation
    :param begin: begin of the result
    :param end: end of the result
    :param region: node specific to plot
    :return:
    """
    result = tools.get_result(path_simulation, begin, end)
    times = result[0][0]
    rateE = result[0][1][:, 0, :]
    stdE = result[0][1][:, 2, :]
    rateI = result[0][1][:, 1, :]
    stdI = result[0][1][:, 4, :]
    corrEI = result[0][1][:, 3, :]
    adaptationE = result[0][1][:, 5, :]
    adaptationI = result[0][1][:, 6, :]
    noise = result[0][1][:, 7, :]
    external_input_excitatory_to_excitatory = result[0][1][:, 8, :]
    external_input_excitatory_to_inhibitory = result[0][1][:, 9, :]
    external_input_inhibitory_to_excitatory = result[0][1][:, 10, :]
    external_input_inhibitory_to_inhibitory = result[0][1][:, 11, :]

    plt.figure()
    plt.plot(times, rateE, label='excitatory')
    plt.plot(times, rateI, label='inhibitatory')
    plt.legend()

    plt.figure()
    plt.plot(times, adaptationE, label='excitatory')
    plt.plot(times, adaptationI, label='inhibitatory')
    plt.legend()

    plt.figure()
    plt.plot(times, rateE[:, 0], label='excitatory')
    plt.plot(times, rateI[:, 0], label='inhibitory')
    plt.legend()

    plt.figure()
    plt.plot(times, noise, label='noise')
    plt.plot(times, external_input_excitatory_to_excitatory, label='external_input_excitatory_to_excitatory')
    plt.plot(times, external_input_excitatory_to_inhibitory, label='external_input_excitatory_to_inhibitory')
    plt.plot(times, external_input_inhibitory_to_excitatory, label='external_input_inhibitory_to_excitatory')
    plt.plot(times, external_input_inhibitory_to_inhibitory, label='external_input_inhibitory_to_inhibitory')
    plt.legend()


    plt.figure()
    plt.plot(times, noise[:, -1], label='noise', alpha=0.2)
    plt.plot(times, external_input_excitatory_to_excitatory[:, -1], '.', label='external_input_excitatory_to_excitatory', alpha=0.2)
    plt.plot(times, external_input_excitatory_to_inhibitory[:, -1], '.', label='external_input_excitatory_to_inhibitory', alpha=0.2)
    plt.plot(times, external_input_inhibitory_to_excitatory[:, -1], '.', label='external_input_inhibitory_to_excitatory', alpha=0.2)
    plt.plot(times, external_input_inhibitory_to_inhibitory[:, -1], '.', label='external_input_inhibitory_to_inhibitory', alpha=0.2)
    plt.legend()

    plt.figure()
    plt.plot(times, rateE[:, region], label='excitatory')
    plt.plot(times, rateI[:, region], label='inhibitory')
    plt.plot(times, external_input_excitatory_to_excitatory[:, region], label='external_input_excitatory_to_excitatory', alpha=0.2)
    plt.plot(times, external_input_excitatory_to_inhibitory[:, region], label='external_input_excitatory_to_inhibitory', alpha=0.2)
    plt.plot(times, external_input_inhibitory_to_excitatory[:, region], label='external_input_inhibitory_to_excitatory', alpha=0.2)
    plt.plot(times, external_input_inhibitory_to_inhibitory[:, region], label='external_input_inhibitory_to_inhibitory', alpha=0.2)
    plt.legend()


if __name__ == "__main__":
    import os

    path = os.path.dirname(os.path.realpath(__file__)) + '/../../simulation/deterministe/rate_7.0/'
    plot_result(path+"/frequency_10.0/", 5000.0, 20000.0)
    plot_result(path+"/frequency_15.0/", 5000.0, 20000.0)
    plt.show()
