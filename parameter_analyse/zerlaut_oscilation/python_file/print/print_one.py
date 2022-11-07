import parameter_analyse.zerlaut_oscilation.python_file.run.tools_simulation as tools
import matplotlib.pyplot as plt

def print_result(path_simulation, begin, end):
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
    plt.plot(times, external_input_excitatory_to_excitatory[:, -1], label='external_input_excitatory_to_excitatory', alpha=0.2)
    plt.plot(times, external_input_excitatory_to_inhibitory[:, -1], label='external_input_excitatory_to_inhibitory', alpha=0.2)
    plt.plot(times, external_input_inhibitory_to_excitatory[:, -1], label='external_input_inhibitory_to_excitatory', alpha=0.2)
    plt.plot(times, external_input_inhibitory_to_inhibitory[:, -1], label='external_input_inhibitory_to_inhibitory', alpha=0.2)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    path = '/home/kusch/Documents/project/Zerlaut/compare_zerlaut/parameter_analyse/zerlaut_oscilation/simulation/deterministe/test/rate_7.0/'
    path += "/frequency_1_42/"
    print_result(path, 0.0, 200.0)
