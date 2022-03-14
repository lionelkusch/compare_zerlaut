import numpy as np
import matplotlib.pyplot as plt


def print_stability(x_origin, f_origin, s_origin, axis_x, axis_y, tolerance=1e-50, color='k', presicion_x=0.01, presicion_y=0.01):
    resample = [0]
    for index, (i, j) in enumerate(zip(x_origin[axis_x, :], x_origin[axis_y, :])):
        if np.abs(x_origin[axis_x, resample[-1]] - i) > presicion_x or np.abs(x_origin[axis_y, resample[-1]] - j) > presicion_y:
            resample.append(index)
    x = x_origin[:, np.array(resample)]
    f = f_origin[:, np.array(resample)]
    L = x[0, :].shape[0]
    curveind = 0
    real_min = np.real(f[:, 0])
    real_min[np.where(np.abs(real_min) < tolerance)] = 0
    evalstart = np.floor(np.sum(np.heaviside(real_min, 0.5)))
    datamateq = np.zeros((4, L))

    for i in range(L):
        real_min = np.real(f[:, i])
        real_min[np.where(np.abs(real_min) < tolerance)] = 0
        evalind = np.floor(np.sum(np.heaviside(real_min, 0.5)))
        if evalstart != evalind:
            curveind = curveind + 1
            evalstart = evalind
        datamateq[0, i] = x[axis_x, i]  # This is the parameter that is varied.
        datamateq[1, i] = x[axis_y, i]  # This is the dependent axis of the bifurcation plot.  The one you wish to plot
        datamateq[2, i] = evalind
        datamateq[3, i] = curveind

    curveindeq = curveind + 1
    curveeq = []
    for i in range(curveindeq):
        index = np.where(datamateq[3, :] == i)[0]
        curveeq.append(datamateq[:3, index])

    for i in range(curveindeq):
        stability = curveeq[i][2, 0]
        if stability == 0:
            plotsty = ''
            plotcolor = 'k'
        else:
            plotsty = '--'
            plotsty_list = ['--', ':', '-.']
            plotsty = plotsty_list[int(np.mod(stability, 6)) - 1]
            c = ['b', 'g', 'm', 'c', 'y', 'r']
            plotcolor = c[int(np.mod(stability, 6)) - 1]

        plotstr = color + plotsty

        line, = plt.plot(curveeq[i][0, :], curveeq[i][1, :], plotstr, linewidth=4)

    xindex = [i['index'] - 1 for i in s_origin]
    xindex = xindex[1:-1]
    for data in s_origin[1:-1]:
        if data['msg'] != 'Neutral Saddle Equilibrium' and data['label'] != 'BV':
            plt.plot(x_origin[axis_x, data['index'] - 1], x_origin[axis_y, data['index'] - 1], 'rx', markersize=10.0)
            plt.text(x_origin[axis_x, data['index'] - 1], x_origin[axis_y, data['index'] - 1], "  " + data['label'])

    # plt.show()
    return line
