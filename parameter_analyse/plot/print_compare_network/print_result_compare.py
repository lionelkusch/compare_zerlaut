#  Copyright 2023 Aix-Marseille Université
# "Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements; and to You under the Apache License, Version 2.0. "
import sqlite3
import matplotlib.pyplot as plt
from numpy import linspace, meshgrid
from matplotlib.ticker import MaxNLocator
from scipy.interpolate import griddata
import numpy as np


def getData(data_base, table_name, list_variable, name_analysis='global', cond=""):
    """
    get data from database
    :param data_base: path of the database
    :param table_name: name of the table
    :param list_variable: variable to get
    :param name_analysis: name of analysis
    :param cond: additional condition
    :return:
    """
    con = sqlite3.connect(data_base, detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES)
    cursor = con.cursor()
    cursor.execute(
        ' SELECT * FROM ( '
        ' SELECT *'
        ' FROM ' + table_name + ' '
                                " WHERE names_population = '" + name_analysis + "'"
        " AND " + list_variable[0][ 'name'] + ' >= ' + str(list_variable[0]['min']) +
        " AND " + list_variable[0]['name'] + ' <= ' + str(list_variable[0]['max']) +
        " AND " + list_variable[1]['name'] + ' >= ' + str(list_variable[1]['min']) +
        " AND " + list_variable[1]['name'] + ' <= ' + str(list_variable[1]['max']) +
        cond +
        " ORDER BY " + list_variable[0]['name'] + ')'
        " ORDER BY " + list_variable[1]['name']
    )
    data_all = cursor.fetchall()
    name_column = [description[0] for description in cursor.description]
    datas = {}
    for id, name in enumerate(name_column):
        datas[name] = []
        for data in data_all:
            datas[name].append(data[id])
    return datas


def grid(x, y, z, res, resX, resY, id=None):
    """
    Convert 3 column data to matplotlib grid

    :param x: x values
    :param y: y values
    :param z: z values
    :param res: boolean if resolution
    :param resX: resolution X
    :param resY: resolution Y
    :param id: selection of result
    :return:
    """
    if res:
        xi = linspace(min(x), max(x), resX)
        yi = linspace(min(y), max(y), resY)
        Z = griddata(x, y, z, xi, yi, interp='linear')
        X, Y = meshgrid(xi, yi)
        return X, Y, Z
    if id is None:
        id = np.where(np.array(z) != None)
    result_x = np.array(x, dtype='float64')[id]
    result_y = np.array(y, dtype='float64')[id]
    result_z = np.array(z, dtype='float64')[id]
    return [result_x, result_y, result_z]


def draw_contour(fig, ax, X, Y, Z, resolution, title, xlabel, ylabel, zlabel, label_size, number_size):
    """
    create graph with contour
    :param fig: figure
    :param ax: axis
    :param X: x values
    :param Y: y values
    :param Z: z values
    :param resolution: boolean if resolution
    :param title: title of the graph
    :param xlabel: x label
    :param ylabel: y label
    :param zlabel: z label
    :param label_size: size of the label
    :param number_size: size of the number
    :return:
    """
    if len(Z) > 4:
        if resolution:
            CS = ax.contourf(X, Y, Z, extend='both')
        else:
            CS = ax.tricontourf(X, Y, Z, extend='both')
        cbar = fig.colorbar(CS, ax=ax)
        cbar.ax.set_ylabel(zlabel, {"fontsize": label_size})
        cbar.ax.tick_params(labelsize=number_size)
        ax.set_xlabel(xlabel, {"fontsize": label_size})
        ax.set_ylabel(ylabel, {"fontsize": label_size})
        ax.set_title(title, {"fontsize": label_size})
    else:
        Warning('not enough values')


def draw_contour_limit(fig, ax, X, Y, Z, resolution, title, xlabel, ylabel, zlabel, zmin, zmax, label_size,
                       number_size, nbins=10, remove_label_y=False):
    """
    create graph with contour with limit
    :param fig: figure where to plot
    :param ax: axis of the graph
    :param X: x values
    :param Y: y values
    :param Z: z values
    :param resolution: boolean if resolution
    :param title: title of the graph
    :param xlabel: x label
    :param ylabel: y label
    :param zlabel: z label
    :param zmin: z minimum values
    :param zmax: z maximum values
    :param label_size: size of the label
    :param number_size: size of the number
    :param nbins: number of bins for the scale
    :param remove_label_y: remove label of y axis
    :return:
    """
    if len(Z) > 3:
        levels = MaxNLocator(nbins=nbins).tick_values(zmin, zmax)
        if resolution:
            CS = ax.contourf(X, Y, Z, resolution, levels=levels, vmin=zmin, vmax=zmax, extend='both')
        else:
            CS = ax.tricontourf(X, Y, Z, resolution, levels=levels, vmin=zmin, vmax=zmax, extend='both')
        cbar = fig.colorbar(CS, ax=ax)
        cbar.ax.set_ylabel(zlabel, {"fontsize": label_size})
        cbar.ax.tick_params(labelsize=number_size)
        ax.set_xlabel(xlabel, {"fontsize": label_size})
        ax.set_ylabel(ylabel, {"fontsize": label_size})
        ax.set_title(title, {"fontsize": label_size})
        if remove_label_y:
            plt.tick_params(
                axis='y',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                right=False,  # ticks along the right edge are off
                left=False,  # ticks along the left edge are off
                labelleft=False,  # labels along the left ticks are off
            )
    else:
        Warning('not enough values')


def draw_point(ax, X, Y, param='w.', size=6.0):
    """
    draw a line or point
    :param ax: axis of graph
    :param X: x values
    :param Y: y values
    :param param: parameters for plotting
    :param size: size of the marker
    :return:
    """
    ax.plot(X, Y, param, markersize=size)


def draw_zone_level(ax, X, Y, Z, resolution, level, color):
    """
    draw a zone of color in transparency
    :param ax: axis of the graph
    :param X: x values
    :param Y: y values
    :param Z: z values
    :param resolution: boolean if resolution
    :param level: level of plotting
    :param color: color of the zone
    :return:
    """
    if len(Z) > 3:
        if np.min(Z) <= level:
            draw_line_level(ax, X, Y, Z, resolution, level, color)
            if resolution:
                ax.contourf(X, Y, Z, levels=[np.min(Z), level], colors=[color], alpha=0.2)
            else:
                ax.tricontourf(X, Y, Z, levels=[np.min(Z), level], colors=[color], alpha=0.2)
    else:
        Warning('not enough values')


def draw_line_level(ax, X, Y, Z, resolution, level, color):
    """
    draw a line
    :param ax: axis of the graph
    :param X: x values
    :param Y: y values
    :param Z: z values
    :param resolution: boolean if resolution
    :param level: level of plotting
    :param color: color of the line
    :return:
    """
    if len(Z) > 3:
        if resolution:
            ax.contour(X, Y, Z, levels=[level], colors=[color], linewidths=2.0)
        else:
            ax.tricontour(X, Y, Z, levels=[level], colors=[color], linewidths=2.0)
    else:
        Warning('not enough values')


def set_lim(ax, ymax, ymin, xmax, xmin, number_size):
    """
    set limit of the values
    :param ax: axis of the graph
    :param ymax: maximum values of y for plotting
    :param ymin: minimum values of y for plotting
    :param xmax: maximum values of x for plotting
    :param xmin: minimum values of x for plotting
    :param number_size: size of the label
    :return:
    """
    ax.set_ylim(ymax=ymax, ymin=ymin)
    ax.set_xlim(xmax=xmax, xmin=xmin)
    ax.tick_params(axis='both', labelsize=number_size)


def min_max(data):
    """
    fix the minimum and maximum of data
    :param data: data
    :return:
    """
    data[np.where(data == None)] = np.NAN
    zmin = np.nanmin(data)
    zmax = np.nanmax(data)
    return [zmin, zmax]


def plot_analysis(data_base_network, data_base_mean_field, table_name_network, table_name_meanfield, list_variable, resX=None, resY=None,
                  label_size=30.0, number_size=20.0, level_percentage=0.95, population='excitatory',
                  measure_network='PLV_w5ms', measure_mean_field='PLV_value',
                  title_measure='PLV w5 ms', unit='PLV', min_value=None, max_value=None, rate=0.0,
                  figsize=(20, 20)
                  ):
    """
    create pdf with all the analysis
    :param data_base_network: database of neural network simulation
    :param data_base_mean_field: database of the mean field
    :param table_name_network: table in the database for the neural network
    :param table_name_meanfield: table in the database for the mean field
    :param list_variable: list of variable to plot
    :param resX: resolution x
    :param resY: resolution y
    :param label_size: size of the label
    :param number_size: size of the number
    :param level_percentage: level of the percentage
    :param population: population of neurons
    :param measure_network: name of the measure for the neural network
    :param measure_mean_field: name of the measure for the mean field
    :param title_measure: title of the measure
    :param unit: unit of the measure
    :param min_value: minimum value
    :param max_value: maximum value
    :param rate: selection of average input
    :param figsize: size of the figure
    """
    name_var1 = list_variable[0]['name']
    name_var2 = list_variable[1]['name']
    title_var1 = list_variable[0]['title']
    title_var2 = list_variable[1]['title']

    if 'disp_max' in list_variable[0].keys():
        xmax = list_variable[0]['disp_max']
    else:
        xmax = None
    if 'disp_min' in list_variable[0].keys():
        xmin = list_variable[0]['disp_min']
    else:
        xmin = None
    if 'disp_max' in list_variable[1].keys():
        ymax = list_variable[1]['disp_max']
    else:
        ymax = None
    if 'disp_min' in list_variable[1].keys():
        ymin = list_variable[1]['disp_min']
    else:
        ymin = None

    data_network = getData(data_base_network, table_name_network, list_variable, population)
    resolution = resX != None and resY != None
    if resX == None:
        resX = len(np.unique(data_network[name_var1]))
    if resY == None:
        resY = len(np.unique(data_network[name_var2]))
    X_network, Y_network, Z_network = grid(data_network[name_var1], data_network[name_var2], data_network[measure_network], res=resolution,
                   resX=resX, resY=resY)
    data_meanfield = getData(data_base_mean_field, table_name_meanfield, list_variable, population, cond=" AND rate == " + str(rate))
    X_mean, Y_mean, Z_mean = grid(data_meanfield[name_var1], data_meanfield[name_var2], data_meanfield[measure_mean_field], res=resolution,
                   resX=resX, resY=resY)
    if max_value is None:
        max_value = np.max(np.concatenate((Z_mean, Z_network)).ravel())
    if min_value is None:
        min_value = np.min(np.concatenate((Z_mean, Z_network)).ravel())

    fig = plt.figure(figsize=figsize)
    #### network
    ax = plt.subplot(1, 2, 1)
    draw_contour_limit(fig, ax, X_network, Y_network, Z_network, resolution, title_measure, title_var1, title_var2, unit, min_value, max_value, label_size, number_size)
    draw_point(ax, X_network, Y_network)
    set_lim(ax, ymax, ymin, xmax, xmin, number_size)
    ax.set_title('network', fontdict={'fontsize': label_size})
    #### mean field
    ax = plt.subplot(1, 2, 2)
    draw_contour_limit(fig, ax, X_mean, Y_mean, Z_mean, resolution, title_measure, title_var1, title_var2, unit, min_value, max_value, label_size, number_size)
    draw_point(ax, X_mean, Y_mean)
    set_lim(ax, ymax, ymin, xmax, xmin, number_size)
    ax.set_title('mean field', fontdict={'fontsize': label_size})
    # plt.show()


if __name__ == "__main__":
    import os
    path_init = os.path.dirname(os.path.realpath(__file__)) + '/../../'
    folder_network = '/simulation/'
    folder_meanfield = '/deterministe/'
    for measure_network, measure_mean_field, title_measure, unit, min_value, max_value in [
        ('PLV_w5ms', 'PLV_value', 'PLV w5 ms', 'PLV', 0.5, None),
        ('MeanPhaseShift_5ms', 'PLV_angle', 'PLV angle w5 ms', 'rad', None, None),
        ('max_IFR_w5ms', 'max_rates', 'max rate of IFR', 'Hz', None, None),
    ]:
        for rate in [0.0, 7.0]:
            for name_population in ['excitatory', 'inhibitory']:
                plot_analysis(
                    path_init+'/spike_oscilation/simulation/'+folder_network+'/rate_'+str(rate)+'/amplitude_frequency.db',
                    path_init+'/zerlaut_oscilation/simulation/'+folder_meanfield+'/database.db',
                    'first_exploration', 'exploration',
                    [
                        {'name': 'amplitude', 'title': 'amplitude ', 'min': 0.0, 'max': 5000000.0},
                        {'name': 'frequency', 'title': 'frequency input', 'min': 0.0, 'max': 5000000.0},
                    ], rate=0.0, population=name_population,
                    measure_network=measure_network, measure_mean_field=measure_mean_field,
                    title_measure=title_measure, unit=unit, min_value=min_value, max_value=max_value,
                    figsize=(20, 10))
                plt.subplots_adjust(top=0.95, bottom=0.1, left=0.05, right=0.97, wspace=0.15, hspace=0.2)
                plt.savefig('figure/'+measure_mean_field+'_'+name_population+'_'+str(rate)+'.pdf')
