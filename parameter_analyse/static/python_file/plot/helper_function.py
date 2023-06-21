from elephant.statistics import isi
import numpy as np

def slidding_window(data, width):
    """
    sliding mean of a time series
    :param data: instantaneous firing rate
    :param width: windows or times average of the mean field
    :return: state variable of the mean field
    """
    res = np.zeros((data.shape[0] - width, width))
    res[:, :] = np.squeeze(data[np.array([[i + j for i in range(width)] for j in range(data.shape[0] - width)])])
    return res.mean(axis=1)


def detection_burst(gids, data, begin, end, limit_burst):
    """
    Compute different measure on the spike
    :param gids: the array with the id
    :param data: the times spikes
    :param begin: the first time
    :param end: the end time
    :param limit_burst: maximum of time for be a burst
    :return:
    """
    nb_neuron = 0
    for gid in gids:
        nb_neuron = nb_neuron + gid[1] - gid[0]
    spikes = []
    list_burst = []
    for j, gid in enumerate(gids):
        for i in np.arange(gid[0], gid[1], 1):
            spike_i = data[1, np.where(data[0, :] == i)]
            spike_i = (spike_i[np.where(spike_i >= begin)]).flatten()
            spike_i = (spike_i[np.where(spike_i <= end)])
            spikes.append(spike_i)
            isis = isi(spike_i)
            isis_add = np.insert(isis, 0, limit_burst + 10.0)
            isis_add = np.append(isis_add, limit_burst + 10.0)
            id_begin_burst = np.where((isis_add > limit_burst) & (np.roll(isis_add, -1) <= limit_burst))[0]
            id_end_burst = np.where((isis_add <= limit_burst) & (np.roll(isis_add, -1) > limit_burst))[0]
            time_start_burst = spike_i[id_begin_burst]
            time_stop_burst = spike_i[id_end_burst]
            nb_burst = id_end_burst - id_begin_burst + 1
            list_burst.append(
                np.array([i * np.ones_like(time_start_burst), time_start_burst, time_stop_burst, nb_burst]))
    if list_burst == []:
        return []
    else:
        return np.concatenate(list_burst, axis=1)


def get_gids_all(path):
    """
    Get the id of the different neurons in the network by population
    :param path: the path to the file
    :return: an array with the range of id and name for the different population
    """
    gidfile = open(path + '/population_GIDs.dat', 'r')
    gids = {}
    for l in gidfile:
        a = l.split()
        try:
            gids[a[2]].append([int(a[0]), int(a[1])])
        except:
            gids[a[2]] = [[int(a[0]), int(a[1])]]
    return gids


def load_spike_all(gids_all, path, begin, end):
    """
    Get the id of the neurons which create the spike
    :param gids_all: dictionary of ids indexes by names
    :param path: the path to the file
    :param begin: the first time
    :param end: the end of time
    :return: The spike of all neurons between end and begin
    """
    data = {}
    data_concatenated = np.concatenate((np.loadtxt(path + "/spike_recorder_ex.dat"),
                                        np.loadtxt(path + "/spike_recorder_in.dat")))
    if data_concatenated.size < 5:
        print('empty file')
        return -1
    data_raw = data_concatenated[np.argsort(data_concatenated[:, 1])]
    idx_time = ((data_raw[:, 1] > begin) * (data_raw[:, 1] < end))
    data_tmp = data_raw[idx_time]
    for name, gids in gids_all.items():
        data[name] = []
        for i in list(range(len(gids))):
            idx_id = ((data_tmp[:, 0] >= gids[i][0]) * (data_tmp[:, 0] <= gids[i][1]))
            data[name].append(data_tmp[idx_id])
        data[name] = np.swapaxes(np.concatenate(data[name], axis=0), 0, 1)
    return data


def load_spike_all_long(path, firing_rate_init, firing_rate_min, increment_firing_rate):
    """
    Get the id of the neurons which create the spike
    :param path: path of the file
    :param firing_rate_init: start of firing rate
    :param firing_rate_min: minimum of firing rate
    :param increment_firing_rate: value of reducing firing rate
    :return: The spike of all neurons between end and begin
    """
    data = {}
    for firing_rate in np.arange(firing_rate_init, firing_rate_min, increment_firing_rate):
        print(firing_rate)
        name_firing_rate = str(np.around(firing_rate))
        data[name_firing_rate] = {'excitatory': [], 'inhibitory': []}
        for name, short_name in [('excitatory', 'ex'), ('inhibitory', 'in')]:
            data_concatenated = np.loadtxt(path + "/"+name_firing_rate+"_spike_recorder_"+short_name+".dat")
            data_raw = data_concatenated[np.argsort(data_concatenated[:, 1])]
            data[name_firing_rate][name] = data_raw
    return data
