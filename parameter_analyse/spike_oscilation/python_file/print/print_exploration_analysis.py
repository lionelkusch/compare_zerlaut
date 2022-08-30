import sqlite3
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from numpy import linspace, meshgrid
from matplotlib.ticker import MaxNLocator
from scipy.interpolate import griddata
import numpy as np


def getData(data_base, table_name, list_variable, name_analysis='global'):
    """
    get data from database
    :param data_base: path of the database
    :param table_name: name of the table
    :param list_variable: variable to get
    :param name_analysis: name of analysis
    :return:
    """
    con = sqlite3.connect(data_base, detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES)
    cursor = con.cursor()
    cursor.execute(
        ' SELECT * FROM ( '
        ' SELECT *'
        ' FROM ' + table_name + ' '
                                " WHERE names_population = '" + name_analysis + "'"
                                                                                " AND " + list_variable[0][
            'name'] + ' >= ' + str(list_variable[0]['min']) +
        " AND " + list_variable[0]['name'] + ' <= ' + str(list_variable[0]['max']) +
        " AND " + list_variable[1]['name'] + ' >= ' + str(list_variable[1]['min']) +
        " AND " + list_variable[1]['name'] + ' <= ' + str(list_variable[1]['max']) +
        " ORDER BY " + list_variable[0]['name'] + ')'
                                                  " ORDER BY " + list_variable[1]['name']
    )
    data_all = cursor.fetchall()
    # datas_select = []
    # for i in data_all:
    #     if i[3] == 0.1 or i[3]%10 ==0:
    #         datas_select.append(i)
    # data_all = datas_select
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


def print_exploration_analysis_pdf(path_figure, data_base, table_name, list_variable, resX=None, resY=None,
                                   label_size=30.0, number_size=20.0, level_percentage=0.95, population='excitatory'):
    """
    create pdf with all the analysis
    :param path_figure: path to save the figure
    :param data_base: path of the database
    :param table_name: name of the table
    :param list_variable: list of variable to plot
    :param resX: resolution x
    :param resY: resolution y
    :param label_size: size of the label
    :param number_size: size of the number
    :param level_percentage: level of the percentage
    """
    name_var1 = list_variable[0]['name']
    name_var2 = list_variable[1]['name']
    title_var1 = list_variable[0]['title']
    title_var2 = list_variable[1]['title']
    data_global = getData(data_base, table_name, list_variable, population)
    resolution = resX != None and resY != None
    if resX == None:
        resX = len(np.unique(data_global[name_var1]))
    if resY == None:
        resY = len(np.unique(data_global[name_var2]))
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

    file = PdfPages(path_figure)

    ## GLOBAL
    ### Analysis
    fig_rate, axs_rate = plt.subplots(2, 2, figsize=(20, 20))
    fig_rate.subplots_adjust(hspace=1.0)

    #### PLV
    ax = axs_rate.ravel()[0]
    X, Y, Z = grid(data_global[name_var1], data_global[name_var2], data_global['PLV_w5ms'], res=resolution,
                   resX=resX, resY=resY)
    draw_contour(fig_rate, ax, X, Y, Z, resolution, 'PLV w5 ms', title_var1, title_var2,
                 'PLV', label_size, number_size)
    draw_point(ax, X, Y)
    set_lim(ax, ymax, ymin, xmax, xmin, number_size)

    #### Angle
    ax = axs_rate.ravel()[1]
    X, Y, Z = grid(data_global[name_var1], data_global[name_var2], data_global['PLV_angle_w5ms'], res=resolution,
                   resX=resX, resY=resY)
    draw_contour(fig_rate, ax, X, Y, Z, resolution, 'PLV angle w5', title_var1,
                 title_var2, 'rad', label_size, number_size)
    draw_point(ax, X, Y)
    set_lim(ax, ymax, ymin, xmax, xmin, number_size)

    #### ISI
    ax = axs_rate.ravel()[2]
    X, Y, Z = grid(data_global[name_var1], data_global[name_var2], data_global['ISI_min'], res=resolution, resX=resX,
                   resY=resY)
    draw_contour(fig_rate, ax, X, Y, Z, resolution, 'minimum of Interspiking interval ', title_var1, title_var2,
                 'ms', label_size, number_size)
    draw_point(ax, X, Y)
    set_lim(ax, ymax, ymin, xmax, xmin, number_size)

    #### timescale
    ax = axs_rate.ravel()[3]
    X, Y, Z = grid(data_global[name_var1], data_global[name_var2], data_global['timescale_w5ms'], res=resolution,
                   resX=resX, resY=resY,
                   id=np.where(np.logical_and(data_global['timescale_w5ms'], data_global['ISI_min'])))
    X1, Y1, Z1 = grid(data_global[name_var1], data_global[name_var2], data_global['ISI_min'], res=resolution,
                      resX=resX, resY=resY, id=np.where(np.logical_and(data_global['timescale_w5ms'], data_global['ISI_min'])))
    draw_line_level(ax, X, Y, Z - Z1, resolution, 0.0, 'red')
    draw_contour_limit(fig_rate, ax, X, Y, Z, resolution, ' times scale w5 ms', title_var1, title_var2,
                       'ms', 5.0, 50.0, label_size, number_size)
    draw_point(ax, X, Y)
    set_lim(ax, ymax, ymin, xmax, xmin, number_size)

    plt.savefig(file, format='pdf')
    plt.close('all')

    ### Analysis
    fig_rate, axs_rate = plt.subplots(2, 2, figsize=(20, 20))
    fig_rate.subplots_adjust(hspace=1.0)

    #### PLV
    ax = axs_rate.ravel()[0]
    X, Y, Z = grid(data_global[name_var1], data_global[name_var2], data_global['PLV_5ms'], res=resolution,
                   resX=resX, resY=resY)
    draw_contour(fig_rate, ax, X, Y, Z, resolution, 'PLV 5 ms', title_var1, title_var2,
                 'PLV', label_size, number_size)
    draw_point(ax, X, Y)
    set_lim(ax, ymax, ymin, xmax, xmin, number_size)

    #### Angle
    ax = axs_rate.ravel()[1]
    X, Y, Z = grid(data_global[name_var1], data_global[name_var2], data_global['PLV_angle_5ms'], res=resolution,
                   resX=resX, resY=resY)
    draw_contour(fig_rate, ax, X, Y, Z, resolution, 'PLV angle 5', title_var1,
                 title_var2, 'rad', label_size, number_size)
    draw_point(ax, X, Y)
    set_lim(ax, ymax, ymin, xmax, xmin, number_size)

    #### ISI
    ax = axs_rate.ravel()[2]
    X, Y, Z = grid(data_global[name_var1], data_global[name_var2], data_global['ISI_min'], res=resolution, resX=resX,
                   resY=resY)
    draw_contour(fig_rate, ax, X, Y, Z, resolution, 'minimum of Interspiking interval ', title_var1, title_var2,
                 'ms', label_size, number_size)
    draw_point(ax, X, Y)
    set_lim(ax, ymax, ymin, xmax, xmin, number_size)

    #### timescale
    ax = axs_rate.ravel()[3]
    X, Y, Z = grid(data_global[name_var1], data_global[name_var2], data_global['timescale_5ms'], res=resolution,
                   resX=resX, resY=resY,
                   id=np.where(np.logical_and(data_global['timescale_5ms'], data_global['ISI_min'])))
    X1, Y1, Z1 = grid(data_global[name_var1], data_global[name_var2], data_global['ISI_min'], res=resolution,
                      resX=resX, resY=resY,
                      id=np.where(np.logical_and(data_global['timescale_5ms'], data_global['ISI_min'])))
    draw_line_level(ax, X, Y, Z - Z1, resolution, 0.0, 'red')
    draw_contour_limit(fig_rate, ax, X, Y, Z, resolution, ' times scale 5 ms', title_var1, title_var2,
                       'ms', 5.0, 50.0, label_size, number_size)
    draw_point(ax, X, Y)
    set_lim(ax, ymax, ymin, xmax, xmin, number_size)

    plt.savefig(file, format='pdf')
    plt.close('all')

    ### Analysis
    fig_rate, axs_rate = plt.subplots(2, 2, figsize=(20, 20))
    fig_rate.subplots_adjust(hspace=1.0)

    #### PLV
    ax = axs_rate.ravel()[0]
    X, Y, Z = grid(data_global[name_var1], data_global[name_var2], data_global['PLV_0_1ms'], res=resolution,
                   resX=resX, resY=resY)
    draw_contour(fig_rate, ax, X, Y, Z, resolution, 'PLV 0.1 ms', title_var1, title_var2,
                 'PLV', label_size, number_size)
    draw_point(ax, X, Y)
    set_lim(ax, ymax, ymin, xmax, xmin, number_size)

    #### Angle
    ax = axs_rate.ravel()[1]
    X, Y, Z = grid(data_global[name_var1], data_global[name_var2], data_global['PLV_angle_0_1ms'], res=resolution,
                   resX=resX, resY=resY)
    draw_contour(fig_rate, ax, X, Y, Z, resolution, 'PLV angle 0.1', title_var1,
                 title_var2, 'rad', label_size, number_size)
    draw_point(ax, X, Y)
    set_lim(ax, ymax, ymin, xmax, xmin, number_size)

    #### ISI
    ax = axs_rate.ravel()[2]
    X, Y, Z = grid(data_global[name_var1], data_global[name_var2], data_global['ISI_min'], res=resolution, resX=resX,
                   resY=resY)
    draw_contour(fig_rate, ax, X, Y, Z, resolution, 'minimum of Interspiking interval ', title_var1, title_var2,
                 'ms', label_size, number_size)
    draw_point(ax, X, Y)
    set_lim(ax, ymax, ymin, xmax, xmin, number_size)

    #### timescale
    ax = axs_rate.ravel()[3]
    X, Y, Z = grid(data_global[name_var1], data_global[name_var2], data_global['timescale_0_1ms'], res=resolution,
                   resX=resX,
                   resY=resY,
                   id=np.where(np.logical_and(data_global['timescale_0_1ms'], data_global['ISI_min'])))
    X1, Y1, Z1 = grid(data_global[name_var1], data_global[name_var2], data_global['ISI_min'], res=resolution,
                      resX=resX, resY=resY,
                   id=np.where(np.logical_and(data_global['timescale_0_1ms'], data_global['ISI_min'])))
    draw_line_level(ax, X, Y, Z - Z1, resolution, 0.0, 'red')
    draw_contour_limit(fig_rate, ax, X, Y, Z, resolution, ' times scale 0.1 ms', title_var1, title_var2,
                       'ms', 5.0, 50.0, label_size, number_size)
    draw_point(ax, X, Y)
    set_lim(ax, ymax, ymin, xmax, xmin, number_size)

    plt.savefig(file, format='pdf')
    plt.close('all')

    ### Analysis
    fig_rate, axs_rate = plt.subplots(2, 2, figsize=(20, 20))
    fig_rate.subplots_adjust(hspace=1.0)

    #### PLV
    ax = axs_rate.ravel()[0]
    X, Y, Z = grid(data_global[name_var1], data_global[name_var2], data_global['PLV_1ms'], res=resolution,
                   resX=resX, resY=resY)
    draw_contour(fig_rate, ax, X, Y, Z, resolution, 'PLV 1 ms', title_var1, title_var2,
                 'PLV', label_size, number_size)
    draw_point(ax, X, Y)
    set_lim(ax, ymax, ymin, xmax, xmin, number_size)

    #### Angle
    ax = axs_rate.ravel()[1]
    X, Y, Z = grid(data_global[name_var1], data_global[name_var2], data_global['PLV_angle_1ms'], res=resolution,
                   resX=resX, resY=resY)
    draw_contour(fig_rate, ax, X, Y, Z, resolution, 'PLV angle', title_var1,
                 title_var2, 'rad', label_size, number_size)
    draw_point(ax, X, Y)
    set_lim(ax, ymax, ymin, xmax, xmin, number_size)

    #### ISI
    ax = axs_rate.ravel()[2]
    X, Y, Z = grid(data_global[name_var1], data_global[name_var2], data_global['ISI_min'], res=resolution, resX=resX,
                   resY=resY)
    draw_contour(fig_rate, ax, X, Y, Z, resolution, 'minimum of Interspiking interval ', title_var1, title_var2,
                 'ms', label_size, number_size)
    draw_point(ax, X, Y)
    set_lim(ax, ymax, ymin, xmax, xmin, number_size)

    #### timescale
    ax = axs_rate.ravel()[3]
    X, Y, Z = grid(data_global[name_var1], data_global[name_var2], data_global['timescale_1ms'], res=resolution,
                   resX=resX,
                   resY=resY,
                   id=np.where(np.logical_and(data_global['timescale_1ms'], data_global['ISI_min'])))
    X1, Y1, Z1 = grid(data_global[name_var1], data_global[name_var2], data_global['ISI_min'], res=resolution,
                      resX=resX, resY=resY,
                   id=np.where(np.logical_and(data_global['timescale_1ms'], data_global['ISI_min'])))
    draw_line_level(ax, X, Y, Z - Z1, resolution, 0.0, 'red')
    draw_contour_limit(fig_rate, ax, X, Y, Z, resolution, ' times scale 1 ms', title_var1, title_var2,
                       'ms', 5.0, 50.0, label_size, number_size)
    draw_point(ax, X, Y)
    set_lim(ax, ymax, ymin, xmax, xmin, number_size)

    plt.savefig(file, format='pdf')
    plt.close('all')

    ### Analysis
    fig_rate, axs_rate = plt.subplots(2, 2, figsize=(20, 20))
    fig_rate.subplots_adjust(hspace=1.0)

    #### max hist 5 ms
    ax = axs_rate.ravel()[0]
    X, Y, Z = grid(data_global[name_var1], data_global[name_var2], data_global['max_IFR_w5ms'], res=resolution,
                   resX=resX, resY=resY)
    draw_contour(fig_rate, ax, X, Y, Z, resolution, ' max IRF w5 ms', title_var1, title_var2,
                 'mean spikes/5ms', label_size, number_size)
    draw_point(ax, X, Y)
    set_lim(ax, ymax, ymin, xmax, xmin, number_size)

    #### Angle
    ax = axs_rate.ravel()[1]
    X, Y, Z = grid(data_global[name_var1], data_global[name_var2], data_global['max_IFR_5ms'], res=resolution,
                   resX=resX, resY=resY)
    draw_contour(fig_rate, ax, X, Y, Z, resolution, ' max IRF 5 ms', title_var1,
                 title_var2, 'spikes/5ms', label_size, number_size)
    draw_point(ax, X, Y)
    set_lim(ax, ymax, ymin, xmax, xmin, number_size)

    #### ISI
    ax = axs_rate.ravel()[2]
    X, Y, Z = grid(data_global[name_var1], data_global[name_var2], data_global['max_IFR_1ms'], res=resolution,
                   resX=resX,
                   resY=resY)
    draw_contour(fig_rate, ax, X, Y, Z, resolution, ' max IRF 1 ms', title_var1, title_var2,
                 'spikes/1ms', label_size, number_size)
    draw_point(ax, X, Y)
    set_lim(ax, ymax, ymin, xmax, xmin, number_size)

    #### timescale
    ax = axs_rate.ravel()[3]
    X, Y, Z = grid(data_global[name_var1], data_global[name_var2], data_global['max_IFR_0_1ms'], res=resolution,
                   resX=resX,
                   resY=resY)
    draw_contour(fig_rate, ax, X, Y, Z, resolution, ' max IRF 0.1 ms', title_var1, title_var2,
                 'spikes/0.1ms', label_size, number_size)
    draw_point(ax, X, Y)
    set_lim(ax, ymax, ymin, xmax, xmin, number_size)

    plt.savefig(file, format='pdf')
    plt.close('all')

    ## GLOBAL
    ### Rate
    fig_rate, axs_rate = plt.subplots(2, 2, figsize=(20, 20))
    fig_rate.subplots_adjust(hspace=1.0)

    #### Mean rate
    ax = axs_rate.ravel()[0]
    X, Y, Z = grid(data_global[name_var1], data_global[name_var2], data_global['rates_average'], res=resolution,
                   resX=resX, resY=resY)
    draw_contour(fig_rate, ax, X, Y, Z, resolution, 'Mean firing rate of the all network', title_var1, title_var2,
                 'Firing rate in Hz', label_size, number_size)
    draw_point(ax, X, Y)
    set_lim(ax, ymax, ymin, xmax, xmin, number_size)

    #### Mean rate
    ax = axs_rate.ravel()[1]
    X, Y, Z = grid(data_global[name_var1], data_global[name_var2], data_global['rates_average'], res=resolution,
                   resX=resX, resY=resY)
    draw_contour_limit(fig_rate, ax, X, Y, Z, resolution, 'Mean firing rate of the all network', title_var1,
                       title_var2, 'Firing rate in Hz', 0.0, 30.0, label_size, number_size)
    draw_point(ax, X, Y)
    set_lim(ax, ymax, ymin, xmax, xmin, number_size)

    #### Max rate
    ax = axs_rate.ravel()[2]
    X, Y, Z = grid(data_global[name_var1], data_global[name_var2], data_global['rates_max'], res=resolution, resX=resX,
                   resY=resY)
    draw_contour(fig_rate, ax, X, Y, Z, resolution, 'Max firing rate of the all network', title_var1, title_var2,
                 'Max firing rate in Hz', label_size, number_size)
    draw_point(ax, X, Y)
    set_lim(ax, ymax, ymin, xmax, xmin, number_size)

    #### Min rate
    ax = axs_rate.ravel()[3]
    X, Y, Z = grid(data_global[name_var1], data_global[name_var2], data_global['rates_min'], res=resolution, resX=resX,
                   resY=resY)
    draw_contour(fig_rate, ax, X, Y, Z, resolution, 'Min firing rate of the all network', title_var1, title_var2,
                 'Min firing rate in Hz', label_size, number_size)
    draw_point(ax, X, Y)
    set_lim(ax, ymax, ymin, xmax, xmin, number_size)

    plt.savefig(file, format='pdf')
    plt.close('all')

    ## GLOBAL
    ### Rate 2
    fig_rate_2, axs_rate_2 = plt.subplots(2, 2, figsize=(20, 20))
    fig_rate_2.subplots_adjust(hspace=1.0)

    #### Mean rate phase
    ax = axs_rate_2.ravel()[0]
    X, Y, Z = grid(data_global[name_var1], data_global[name_var2], data_global['frequency_phase_freq'], res=resolution,
                   resX=resX, resY=resY)
    draw_contour(fig_rate_2, ax, X, Y, Z, resolution, 'Frequency of mean phase rate of the all network', title_var1,
                 title_var2, 'Firing rate in Hz', label_size, number_size)
    draw_point(ax, X, Y)
    set_lim(ax, ymax, ymin, xmax, xmin, number_size)

    #### Mean rate hist
    ax = axs_rate_2.ravel()[1]
    X, Y, Z = grid(data_global[name_var1], data_global[name_var2], data_global['frequency_phase_val'], res=resolution,
                   resX=resX, resY=resY)
    draw_contour(fig_rate_2, ax, X, Y, Z, resolution, 'Power of frequency of mean phase of the all network',
                 title_var1, title_var2, 'power spectrum', label_size, number_size)
    draw_point(ax, X, Y)
    set_lim(ax, ymax, ymin, xmax, xmin, number_size)

    #### Max rate
    ax = axs_rate_2.ravel()[2]
    X, Y, Z = grid(data_global[name_var1], data_global[name_var2], data_global['frequency_hist_1_freq'], res=resolution,
                   resX=resX, resY=resY)
    draw_contour(fig_rate_2, ax, X, Y, Z, resolution, 'Frequency rate of the all network', title_var1, title_var2,
                 'Firing rate in Hz', label_size, number_size)
    draw_point(ax, X, Y)
    set_lim(ax, ymax, ymin, xmax, xmin, number_size)

    #### Min rate
    ax = axs_rate_2.ravel()[3]
    X, Y, Z = grid(data_global[name_var1], data_global[name_var2], data_global['frequency_hist_1_val'], res=resolution,
                   resX=resX, resY=resY)
    draw_contour(fig_rate_2, ax, X, Y, Z, resolution, 'Power of frequence hist of the all network', title_var1,
                 title_var2, 'Power spectrum', label_size, number_size)
    draw_point(ax, X, Y)
    set_lim(ax, ymax, ymin, xmax, xmin, number_size)

    plt.savefig(file, format='pdf')
    plt.close('all')

    ### Synchronize
    fig_synch, axs_synch = plt.subplots(2, 2, figsize=(20, 20))
    fig_synch.subplots_adjust(hspace=1.0)

    #### CV_IFR_1ms
    ax = axs_synch.ravel()[0]
    X, Y, Z = grid(data_global[name_var1], data_global[name_var2], data_global['cvs_IFR_1ms'], res=resolution,
                   resX=resX, resY=resY)
    draw_line_level(ax, X, Y, Z, resolution, 1.0, 'red')
    draw_contour(fig_synch, ax, X, Y, Z, resolution, 'Cv of instantaneous firing rate\n with bins = 1ms', title_var1,
                 title_var2, 'Cv IFR', label_size, number_size)
    # draw_contour_limit(fig_synch,ax,X,Y,Z,resolution,'Cv of instantaneous firing rate\n with bins = 1ms',title_var1,title_var2,'Cv IFR',0.0,2.0,label_size,number_size)
    draw_point(ax, X, Y)
    set_lim(ax, ymax, ymin, xmax, xmin, number_size)

    #### CV_IFR_5ms
    ax = axs_synch.ravel()[1]
    X, Y, Z = grid(data_global[name_var1], data_global[name_var2], data_global['cvs_IFR_5ms'], res=resolution,
                   resX=resX, resY=resY)
    draw_line_level(ax, X, Y, Z, resolution, 1.0, 'red')
    draw_contour(fig_synch, ax, X, Y, Z, resolution, 'Cv of instantaneous firing rate\n with bins = 5ms', title_var1,
                 title_var2, 'Cv IFR', label_size, number_size)
    # draw_contour_limit(fig_synch,ax,X,Y,Z,resolution,'Cv of instantaneous firing rate\n with bins = 3ms',title_var1,title_var2,'Cv IFR',0.0,2.0,label_size,number_size)
    draw_point(ax, X, Y)
    set_lim(ax, ymax, ymin, xmax, xmin, number_size)

    #### R_synch
    ax = axs_synch.ravel()[2]
    X, Y, Z = grid(data_global[name_var1], data_global[name_var2], data_global['synch_Rs_average'], res=resolution,
                   resX=resX, resY=resY)
    draw_contour(fig_synch, ax, X, Y, Z, resolution, 'Mean R synchronization', title_var1, title_var2,
                 'R synchronization', label_size, number_size)
    # draw_contour_limit(fig_synch,ax,X,Y,Z,resolution,'Mean R synchronization',title_var1,title_var2,'R synchronization',0.5,1.0,label_size,number_size)
    X_more, Y_more, Z_more = grid(data_global[name_var1], data_global[name_var2], data_global['cvs_ISI_average'],
                                  res=resolution, resX=resX, resY=resY)
    draw_zone_level(ax, X_more, Y_more, Z_more, resolution, 0.5, 'blue')
    X_more, Y_more, Z_more = grid(data_global[name_var1], data_global[name_var2], data_global['percentage'],
                                  res=resolution, resX=resX, resY=resY)
    draw_zone_level(ax, X_more, Y_more, Z_more, resolution, level_percentage, 'red')
    draw_point(ax, X, Y)
    set_lim(ax, ymax, ymin, xmax, xmin, number_size)

    plt.savefig(file, format='pdf')
    plt.close('all')

    ### Synchronization 2
    fig_reg, axs_sync_2 = plt.subplots(2, 2, figsize=(20, 20))
    fig_reg.subplots_adjust(hspace=1.0)

    #### R_synch min
    ax = axs_sync_2.ravel()[0]
    X, Y, Z = grid(data_global[name_var1], data_global[name_var2], data_global['synch_Rs_min'], res=resolution,
                   resX=resX, resY=resY)
    draw_contour(fig_synch, ax, X, Y, Z, resolution, 'Min R synchronization', title_var1, title_var2,
                 'R synchronization', label_size, number_size)
    X_more, Y_more, Z_more = grid(data_global[name_var1], data_global[name_var2], data_global['percentage'],
                                  res=resolution, resX=resX, resY=resY)
    draw_zone_level(ax, X_more, Y_more, Z_more, resolution, level_percentage, 'red')
    draw_point(ax, X, Y)
    set_lim(ax, ymax, ymin, xmax, xmin, number_size)

    #### R_synch max
    ax = axs_sync_2.ravel()[1]
    X, Y, Z = grid(data_global[name_var1], data_global[name_var2], data_global['synch_Rs_max'], res=resolution,
                   resX=resX, resY=resY)
    draw_contour(fig_synch, ax, X, Y, Z, resolution, 'Max R synchronization', title_var1, title_var2,
                 'R synchronization', label_size, number_size)
    X_more, Y_more, Z_more = grid(data_global[name_var1], data_global[name_var2], data_global['percentage'],
                                  res=resolution, resX=resX, resY=resY)
    draw_zone_level(ax, X_more, Y_more, Z_more, resolution, level_percentage, 'red')
    draw_point(ax, X, Y)
    set_lim(ax, ymax, ymin, xmax, xmin, number_size)

    #### R_synch std
    ax = axs_sync_2.ravel()[2]
    X, Y, Z = grid(data_global[name_var1], data_global[name_var2], data_global['synch_Rs_std'], res=resolution,
                   resX=resX, resY=resY)
    draw_contour(fig_synch, ax, X, Y, Z, resolution, 'Variation R synchronization', title_var1, title_var2,
                 'R synchronization', label_size, number_size)
    # draw_zone_level(ax,X,Y,Z,resolution,0.015,'green')
    X_more, Y_more, Z_more = grid(data_global[name_var1], data_global[name_var2], data_global['percentage'],
                                  res=resolution, resX=resX, resY=resY)
    draw_zone_level(ax, X_more, Y_more, Z_more, resolution, level_percentage, 'red')
    draw_point(ax, X, Y)
    set_lim(ax, ymax, ymin, xmax, xmin, number_size)

    #### R_synch cv
    ax = axs_sync_2.ravel()[3]
    X, Y, Z = grid(data_global[name_var1], data_global[name_var2],
                   np.divide(data_global['synch_Rs_std'], data_global['synch_Rs_average'],
                             where=np.array(data_global['synch_Rs_average']) != None), res=resolution, resX=resX,
                   resY=resY)
    draw_contour(fig_synch, ax, X, Y, Z, resolution, 'Coefficient of Variation R synchronization', title_var1,
                 title_var2, 'R synchronization', label_size, number_size)
    X_more, Y_more, Z_more = grid(data_global[name_var1], data_global[name_var2], data_global['percentage'],
                                  res=resolution, resX=resX, resY=resY)
    draw_zone_level(ax, X_more, Y_more, Z_more, resolution, level_percentage, 'red')
    draw_point(ax, X, Y)
    set_lim(ax, ymax, ymin, xmax, xmin, number_size)

    plt.savefig(file, format='pdf')
    plt.close('all')

    ### Regularity
    fig_reg, axs_reg = plt.subplots(2, 2, figsize=(20, 20))
    fig_reg.subplots_adjust(hspace=1.0)

    #### CV_ISI
    ax = axs_reg.ravel()[0]
    X, Y, Z = grid(data_global[name_var1], data_global[name_var2], data_global['cvs_ISI_average'], res=resolution,
                   resX=resX, resY=resY)
    draw_line_level(ax, X, Y, Z, resolution, 0.2, 'blue')
    draw_contour(fig_reg, ax, X, Y, Z, resolution, 'Cv of interspiking intervalle', title_var1, title_var2, 'Cv ISI',
                 label_size, number_size)
    X_more, Y_more, Z_more = grid(data_global[name_var1], data_global[name_var2], data_global['percentage'],
                                  res=resolution, resX=resX, resY=resY)
    draw_zone_level(ax, X_more, Y_more, Z_more, resolution, level_percentage, 'red')
    draw_point(ax, X, Y)
    set_lim(ax, ymax, ymin, xmax, xmin, number_size)

    #### LV_ISI
    ax = axs_reg.ravel()[1]
    X, Y, Z = grid(data_global[name_var1], data_global[name_var2], data_global['lvs_ISI_average'], res=resolution,
                   resX=resX, resY=resY)
    draw_contour(fig_reg, ax, X, Y, Z, resolution, 'Lv of interspiking intervalle', title_var1, title_var2, 'Lv ISI',
                 label_size, number_size)
    X_more, Y_more, Z_more = grid(data_global[name_var1], data_global[name_var2], data_global['percentage'],
                                  res=resolution, resX=resX, resY=resY)
    draw_zone_level(ax, X_more, Y_more, Z_more, resolution, level_percentage, 'red')
    draw_point(ax, X, Y)
    set_lim(ax, ymax, ymin, xmax, xmin, number_size)

    #### Percentage
    ax = axs_reg.ravel()[2]
    X, Y, Z = grid(data_global[name_var1], data_global[name_var2], data_global['percentage'], res=resolution, resX=resX,
                   resY=resY)
    draw_line_level(ax, X, Y, Z, resolution, level_percentage, 'red')
    X, Y, Z = grid(data_global[name_var1], data_global[name_var2], data_global['percentage'], res=resolution, resX=resX,
                   resY=resY)
    draw_contour_limit(fig_reg, ax, X, Y, Z, resolution, 'Percentage of analysed neuron', title_var1, title_var2,
                       'percentage', 0.0, 1.0, label_size, number_size)
    draw_point(ax, X, Y)
    set_lim(ax, ymax, ymin, xmax, xmin, number_size)

    plt.savefig(file, format='pdf')
    plt.close('all')

    ### ISI
    fig_rate, axs_rate = plt.subplots(2, 2, figsize=(20, 20))
    fig_rate.subplots_adjust(hspace=1.0)

    #### Mean ISI
    ax = axs_rate.ravel()[0]
    X, Y, Z = grid(data_global[name_var1], data_global[name_var2], data_global['ISI_average'], res=resolution,
                   resX=resX, resY=resY)
    draw_contour(fig_rate, ax, X, Y, Z, resolution, 'Mean of Inter-Spiking interval\n of the all network', title_var1,
                 title_var2, 'ISI in ms', label_size, number_size)
    draw_point(ax, X, Y)
    set_lim(ax, ymax, ymin, xmax, xmin, number_size)

    #### Std ISI
    ax = axs_rate.ravel()[1]
    X, Y, Z = grid(data_global[name_var1], data_global[name_var2], data_global['ISI_std'], res=resolution, resX=resX,
                   resY=resY)
    draw_contour(fig_rate, ax, X, Y, Z, resolution, 'Std Inter-Spiking interval\n of the all network', title_var1,
                 title_var2, 'ISI in ms', label_size, number_size)
    draw_point(ax, X, Y)
    set_lim(ax, ymax, ymin, xmax, xmin, number_size)

    #### Max ISI
    ax = axs_rate.ravel()[2]
    X, Y, Z = grid(data_global[name_var1], data_global[name_var2], data_global['ISI_max'], res=resolution, resX=resX,
                   resY=resY)
    draw_contour(fig_rate, ax, X, Y, Z, resolution, 'Max Inter-Spiking interval\n of the all network', title_var1,
                 title_var2, 'Max ISI in ms', label_size, number_size)
    draw_point(ax, X, Y)
    set_lim(ax, ymax, ymin, xmax, xmin, number_size)

    #### Min ISI
    ax = axs_rate.ravel()[3]
    X, Y, Z = grid(data_global[name_var1], data_global[name_var2], data_global['ISI_min'], res=resolution, resX=resX,
                   resY=resY)
    draw_line_level(ax, X, Y, Z, resolution, 11.0, 'blue')
    draw_line_level(ax, X, Y, Z, resolution, 0.1, 'red')
    draw_contour(fig_rate, ax, X, Y, Z, resolution, 'Min Inter-Spiking interval\nof the all network', title_var1,
                 title_var2, 'Min ISI in ms', label_size, number_size)
    draw_point(ax, X, Y)
    set_lim(ax, ymax, ymin, xmax, xmin, number_size)

    plt.savefig(file, format='pdf')
    plt.close('all')

    ## Global Burst
    ### Burst Rate
    fig_rate, axs_rate = plt.subplots(2, 2, figsize=(20, 20))
    fig_rate.subplots_adjust(hspace=1.0)

    #### Mean Burst rate
    ax = axs_rate.ravel()[0]
    X, Y, Z = grid(data_global[name_var1], data_global[name_var2], data_global['burst_rate_average'], res=resolution,
                   resX=resX, resY=resY)
    draw_contour(fig_rate, ax, X, Y, Z, resolution, 'Mean burst firing rate of the all network', title_var1,
                 title_var2, 'Firing rate in Hz', label_size, number_size)
    draw_point(ax, X, Y)
    set_lim(ax, ymax, ymin, xmax, xmin, number_size)

    #### Mean Burst rate
    ax = axs_rate.ravel()[1]
    X, Y, Z = grid(data_global[name_var1], data_global[name_var2], data_global['burst_rate_average'], res=resolution,
                   resX=resX, resY=resY)
    draw_contour_limit(fig_rate, ax, X, Y, Z, resolution, 'Mean burst firing rate of the all network', title_var1,
                       title_var2, 'Firing rate in Hz', 0.0, 30.0, label_size, number_size)
    draw_point(ax, X, Y)
    set_lim(ax, ymax, ymin, xmax, xmin, number_size)

    #### Max Burst rate
    ax = axs_rate.ravel()[2]
    X, Y, Z = grid(data_global[name_var1], data_global[name_var2], data_global['burst_rate_max'], res=resolution,
                   resX=resX, resY=resY)
    draw_contour(fig_rate, ax, X, Y, Z, resolution, 'Max burst firing rate of the all network', title_var1, title_var2,
                 'Max firing rate in Hz', label_size, number_size)
    draw_point(ax, X, Y)
    set_lim(ax, ymax, ymin, xmax, xmin, number_size)

    #### Min Burst rate
    ax = axs_rate.ravel()[3]
    X, Y, Z = grid(data_global[name_var1], data_global[name_var2], data_global['burst_rate_min'], res=resolution,
                   resX=resX, resY=resY)
    draw_contour(fig_rate, ax, X, Y, Z, resolution, 'Min burst firing rate of the all network', title_var1, title_var2,
                 'Min firing rate in Hz', label_size, number_size)
    draw_point(ax, X, Y)
    set_lim(ax, ymax, ymin, xmax, xmin, number_size)

    plt.savefig(file, format='pdf')
    plt.close('all')

    ### Nb Burst
    fig_rate, axs_rate = plt.subplots(2, 2, figsize=(20, 20))
    fig_rate.subplots_adjust(hspace=1.0)

    #### Mean Nb Burst
    ax = axs_rate.ravel()[0]
    X, Y, Z = grid(data_global[name_var1], data_global[name_var2], data_global['burst_count_average'], res=resolution,
                   resX=resX, resY=resY)
    draw_contour(fig_rate, ax, X, Y, Z, resolution, 'Mean number of spike in burst of the all network', title_var1,
                 title_var2, 'Number of spikes', label_size, number_size)
    draw_point(ax, X, Y)
    set_lim(ax, ymax, ymin, xmax, xmin, number_size)

    #### Std Nb Burst
    ax = axs_rate.ravel()[1]
    X, Y, Z = grid(data_global[name_var1], data_global[name_var2], data_global['burst_count_std'], res=resolution,
                   resX=resX, resY=resY)
    draw_contour(fig_rate, ax, X, Y, Z, resolution, 'Std number of spike in burst\n of the all network', title_var1,
                 title_var2, 'Number of spikes', label_size, number_size)
    draw_point(ax, X, Y)
    set_lim(ax, ymax, ymin, xmax, xmin, number_size)

    #### Max Nb Burst
    ax = axs_rate.ravel()[2]
    X, Y, Z = grid(data_global[name_var1], data_global[name_var2], data_global['burst_count_max'], res=resolution,
                   resX=resX, resY=resY)
    draw_contour(fig_rate, ax, X, Y, Z, resolution, 'Max number of spike in burst\n of the all network', title_var1,
                 title_var2, 'Max number of spikes', label_size, number_size)
    draw_point(ax, X, Y)
    set_lim(ax, ymax, ymin, xmax, xmin, number_size)

    #### Min Nb Burst
    ax = axs_rate.ravel()[3]
    X, Y, Z = grid(data_global[name_var1], data_global[name_var2], data_global['burst_count_min'], res=resolution,
                   resX=resX, resY=resY)
    draw_contour(fig_rate, ax, X, Y, Z, resolution, 'Min number of spikes in burst\n of the all network', title_var1,
                 title_var2, 'Min number of spikes', label_size, number_size)
    draw_point(ax, X, Y)
    set_lim(ax, ymax, ymin, xmax, xmin, number_size)

    plt.savefig(file, format='pdf')
    plt.close('all')

    ### Time Burst
    fig_rate, axs_rate = plt.subplots(2, 2, figsize=(20, 20))
    fig_rate.subplots_adjust(hspace=1.0)

    #### Mean Time of Burst
    ax = axs_rate.ravel()[0]
    X, Y, Z = grid(data_global[name_var1], data_global[name_var2], data_global['burst_interval_average'],
                   res=resolution, resX=resX, resY=resY)
    draw_contour(fig_rate, ax, X, Y, Z, resolution, 'Mean time of burst of the all network', title_var1, title_var2,
                 'Time of burst in ms', label_size, number_size)
    draw_point(ax, X, Y)
    set_lim(ax, ymax, ymin, xmax, xmin, number_size)

    #### Std time Burst
    ax = axs_rate.ravel()[1]
    X, Y, Z = grid(data_global[name_var1], data_global[name_var2], data_global['burst_interval_std'], res=resolution,
                   resX=resX, resY=resY)
    draw_contour(fig_rate, ax, X, Y, Z, resolution, 'Std time of burst of the all network', title_var1, title_var2,
                 'Time of burst in ms', label_size, number_size)
    draw_point(ax, X, Y)
    set_lim(ax, ymax, ymin, xmax, xmin, number_size)

    #### Max time Burst
    ax = axs_rate.ravel()[2]
    X, Y, Z = grid(data_global[name_var1], data_global[name_var2], data_global['burst_interval_max'], res=resolution,
                   resX=resX, resY=resY)
    draw_contour(fig_rate, ax, X, Y, Z, resolution, 'Max time of burst of the all network', title_var1, title_var2,
                 'Max time of burst in ms', label_size, number_size)
    draw_point(ax, X, Y)
    set_lim(ax, ymax, ymin, xmax, xmin, number_size)

    #### Min time Burst
    ax = axs_rate.ravel()[3]
    X, Y, Z = grid(data_global[name_var1], data_global[name_var2], data_global['burst_interval_min'], res=resolution,
                   resX=resX, resY=resY)
    draw_contour(fig_rate, ax, X, Y, Z, resolution, 'Min time of burst of the all network', title_var1, title_var2,
                 'Min time of burst in ms', label_size, number_size)
    draw_point(ax, X, Y)
    set_lim(ax, ymax, ymin, xmax, xmin, number_size)

    plt.savefig(file, format='pdf')
    plt.close('all')

    ### Burst Regularity
    fig_reg, axs_reg = plt.subplots(2, 2, figsize=(20, 20))
    fig_reg.subplots_adjust(hspace=1.0)

    #### Begin CV_ISI
    ax = axs_reg.ravel()[0]
    X, Y, Z = grid(data_global[name_var1], data_global[name_var2], data_global['burst_cv_begin_average'],
                   res=resolution, resX=resX, resY=resY)
    draw_line_level(ax, X, Y, Z, resolution, 0.2, 'blue')
    draw_contour(fig_reg, ax, X, Y, Z, resolution, 'Cv of interspiking intervalle begin burst', title_var1, title_var2,
                 'Cv ISI', label_size, number_size)
    X_more, Y_more, Z_more = grid(data_global[name_var1], data_global[name_var2], data_global['percentage_burst'],
                                  res=resolution, resX=resX, resY=resY)
    draw_zone_level(ax, X_more, Y_more, Z_more, resolution, level_percentage, 'red')
    draw_point(ax, X, Y)
    set_lim(ax, ymax, ymin, xmax, xmin, number_size)

    #### Begin LV_ISI
    ax = axs_reg.ravel()[1]
    X, Y, Z = grid(data_global[name_var1], data_global[name_var2], data_global['burst_lv_begin_average'],
                   res=resolution, resX=resX, resY=resY)
    draw_contour(fig_reg, ax, X, Y, Z, resolution, 'Lv of interspiking intervalle', title_var1, title_var2, 'Lv ISI',
                 label_size, number_size)
    X_more, Y_more, Z_more = grid(data_global[name_var1], data_global[name_var2], data_global['percentage_burst'],
                                  res=resolution, resX=resX, resY=resY)
    draw_zone_level(ax, X_more, Y_more, Z_more, resolution, level_percentage, 'red')
    draw_point(ax, X, Y)
    set_lim(ax, ymax, ymin, xmax, xmin, number_size)

    #### End CV_ISI
    ax = axs_reg.ravel()[2]
    X, Y, Z = grid(data_global[name_var1], data_global[name_var2], data_global['burst_cv_end_average'], res=resolution,
                   resX=resX, resY=resY)
    draw_line_level(ax, X, Y, Z, resolution, 0.2, 'blue')
    draw_contour(fig_reg, ax, X, Y, Z, resolution, 'Cv of interspiking intervalle end burst', title_var1, title_var2,
                 'Cv ISI', label_size, number_size)
    X_more, Y_more, Z_more = grid(data_global[name_var1], data_global[name_var2], data_global['percentage_burst'],
                                  res=resolution, resX=resX, resY=resY)
    draw_zone_level(ax, X_more, Y_more, Z_more, resolution, level_percentage, 'red')
    draw_point(ax, X, Y)
    set_lim(ax, ymax, ymin, xmax, xmin, number_size)

    #### End LV_ISI
    ax = axs_reg.ravel()[3]
    X, Y, Z = grid(data_global[name_var1], data_global[name_var2], data_global['burst_lv_end_average'], res=resolution,
                   resX=resX, resY=resY)
    draw_contour(fig_reg, ax, X, Y, Z, resolution, 'Lv of interspiking intervalle', title_var1, title_var2, 'Lv ISI',
                 label_size, number_size)
    X_mor, Y_more, Z_more = grid(data_global[name_var1], data_global[name_var2], data_global['percentage_burst'],
                                 res=resolution, resX=resX, resY=resY)
    draw_zone_level(ax, X_more, Y_more, Z_more, resolution, level_percentage, 'red')
    draw_point(ax, X, Y)
    set_lim(ax, ymax, ymin, xmax, xmin, number_size)

    plt.savefig(file, format='pdf')
    plt.close('all')

    # Burst percentage

    fig_reg, axs_reg = plt.subplots(2, 2, figsize=(20, 20))
    fig_reg.subplots_adjust(hspace=1.0)

    #### Percentage
    ax = axs_reg.ravel()[0]
    X, Y, Z = grid(data_global[name_var1], data_global[name_var2], data_global['percentage_burst'], res=resolution,
                   resX=resX, resY=resY)
    draw_line_level(ax, X, Y, Z, resolution, level_percentage, 'red')
    X, Y, Z = grid(data_global[name_var1], data_global[name_var2], data_global['percentage_burst'], res=resolution,
                   resX=resX, resY=resY)
    draw_contour_limit(fig_reg, ax, X, Y, Z, resolution, 'Burst Percentage of analysed neuron', title_var1, title_var2,
                       'percentage', 0.0, 1.0, label_size, number_size)
    draw_point(ax, X, Y)
    set_lim(ax, ymax, ymin, xmax, xmin, number_size)

    plt.savefig(file, format='pdf')
    plt.close('all')

    file.close()


if __name__ == "__main__":
    # folder = 'simulation_3'
    # folder = 'simulation_rate_2.5'
    # folder = 'simulation_rate_7.0'
    folder = 'simulation_rate_amplitude'
    for name_population in ['excitatory', 'inhibitory']:
        print_exploration_analysis_pdf(
            '/home/kusch/Documents/project/Zerlaut/compare_zerlaut/parameter_analyse/spike_oscilation/simulation/'+folder+'/figure/figure_test_' + str(
                name_population) + '.pdf',
            '/home/kusch/Documents/project/Zerlaut/compare_zerlaut/parameter_analyse/spike_oscilation/simulation/'+folder+'/amplitude_frequency.db',
            'first_exploration',
            [
                {'name': 'amplitude', 'title': 'amplitude ', 'min': 0.0, 'max': 500000.0},
                {'name': 'frequency', 'title': 'frequency input', 'min': 0.0, 'max': 5000000.0},
            ], population=name_population)
        # print_exploration_analysis_pdf(
        #     '/home/kusch/Documents/project/Zerlaut/compare_zerlaut/parameter_analyse/spike_oscilation/simulation/'+folder+'/figure/figure_test_2' + str(
        #         name_population) + '.pdf',
        #     '/home/kusch/Documents/project/Zerlaut/compare_zerlaut/parameter_analyse/spike_oscilation/simulation/'+folder+'/amplitude_frequency.db',
        #     'first_exploration',
        #     [
        #         {'name': 'amplitude', 'title': 'amplitude ', 'min': 0.0, 'max': 1.5},
        #         {'name': 'frequency', 'title': 'frequency input', 'min': 0.0, 'max': 5000000.0},
        #     ], population=name_population)
