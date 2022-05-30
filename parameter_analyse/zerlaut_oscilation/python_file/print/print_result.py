import sqlite3
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from numpy import linspace, meshgrid
from matplotlib.ticker import MaxNLocator
from scipy.interpolate import griddata
import numpy as np


def getData(data_base, table_name, list_variable, name_analysis='global', noise=0.0):
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
        " AND " + list_variable[0]['name'] + ' >= ' + str(list_variable[0]['min']) +
        " AND " + list_variable[0]['name'] + ' <= ' + str(list_variable[0]['max']) +
        " AND " + list_variable[1]['name'] + ' >= ' + str(list_variable[1]['min']) +
        " AND " + list_variable[1]['name'] + ' <= ' + str(list_variable[1]['max']) +
        " AND noise = " + str(noise) +
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


def grid(x, y, z, res, resX, resY):
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
                axis='y',       # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                right=False,       # ticks along the right edge are off
                left=False,        # ticks along the left edge are off
                labelleft=False,   # labels along the left ticks are off
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
                                   label_size=30.0, number_size=20.0, level_percentage=0.95, population= 'excitatory',
                                   noise=0.0):
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
    data_global = getData(data_base, table_name, list_variable, population, noise=noise)
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
    X, Y, Z = grid(data_global[name_var1], data_global[name_var2], data_global['PLV_value'], res=resolution,
                   resX=resX, resY=resY)
    draw_contour(fig_rate, ax, X, Y, Z, resolution, 'PLV', title_var1, title_var2,
                 'PLV', label_size, number_size)
    draw_point(ax, X, Y)
    set_lim(ax, ymax, ymin, xmax, xmin, number_size)

    #### Angle
    ax = axs_rate.ravel()[1]
    X, Y, Z = grid(data_global[name_var1], data_global[name_var2], data_global['PLV_angle'], res=resolution,
                   resX=resX, resY=resY)
    draw_contour(fig_rate, ax, X, Y, Z, resolution, 'PLV angle', title_var1,
                       title_var2, 'rad', label_size, number_size)
    draw_point(ax, X, Y)
    set_lim(ax, ymax, ymin, xmax, xmin, number_size)

    #### ISI
    ax = axs_rate.ravel()[2]
    X, Y, Z = grid(data_global[name_var1], data_global[name_var2], data_global['frequency_dom'], res=resolution, resX=resX,
                   resY=resY)
    draw_contour(fig_rate, ax, X, Y, Z, resolution, ' frequency dominant ', title_var1, title_var2,
                 'ms', label_size, number_size)
    draw_point(ax, X, Y)
    set_lim(ax, ymax, ymin, xmax, xmin, number_size)

    plt.savefig(file, format='pdf')
    plt.close('all')


    ## GLOBAL
    ### Analysis
    fig_rate, axs_rate = plt.subplots(2, 2, figsize=(20, 20))
    fig_rate.subplots_adjust(hspace=1.0)

    #### PLV
    ax = axs_rate.ravel()[0]
    X, Y, Z = grid(data_global[name_var1], data_global[name_var2], data_global['max_rates'], res=resolution,
                   resX=resX, resY=resY)
    draw_contour(fig_rate, ax, X, Y, Z, resolution, 'max rate', title_var1, title_var2,
                 'Hz', label_size, number_size)
    draw_point(ax, X, Y)
    set_lim(ax, ymax, ymin, xmax, xmin, number_size)

    #### Angle
    ax = axs_rate.ravel()[1]
    X, Y, Z = grid(data_global[name_var1], data_global[name_var2], data_global['min_rates'], res=resolution,
                   resX=resX, resY=resY)
    draw_contour(fig_rate, ax, X, Y, Z, resolution, 'min rate', title_var1,
                 title_var2, 'Hz', label_size, number_size)
    draw_point(ax, X, Y)
    set_lim(ax, ymax, ymin, xmax, xmin, number_size)

    #### ISI
    ax = axs_rate.ravel()[2]
    X, Y, Z = grid(data_global[name_var1], data_global[name_var2], data_global['mean_rates'], res=resolution, resX=resX,
                   resY=resY)
    draw_contour(fig_rate, ax, X, Y, Z, resolution, ' mean rates', title_var1, title_var2,
                 'Hz', label_size, number_size)
    draw_point(ax, X, Y)
    set_lim(ax, ymax, ymin, xmax, xmin, number_size)

    plt.savefig(file, format='pdf')
    plt.close('all')
    file.close()

if __name__ == "__main__":
    path_root = '/home/kusch/Documents/project/Zerlaut/compare_zerlaut/parameter_analyse/zerlaut_oscilation/simulation/'
    database = path_root + "/database_2.db"
    # for noise in np.arange(1e-9, 1e-8, 1e-9):
    # for noise in np.arange(1e-8, 1e-7, 1e-8):
    for noise in np.arange(0.0, 1e-5, 5e-7):
        for name_population in ['excitatory', 'inhibitory']:
            print_exploration_analysis_pdf(path_root+'/figure/figure_test_'+str(name_population)+'_'+str(noise)+'.pdf',
                                           database,
                                           'exploration',
                                           [
                    {'name': 'amplitude', 'title': 'amplitude ', 'min': 0., 'max': 500000.0},
                   {'name': 'frequency', 'title': 'frequency input', 'min': 0.0, 'max': 5000000.0},
                                           ], population=name_population, noise=noise)