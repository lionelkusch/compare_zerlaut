import matplotlib.pyplot as plt
import numpy as np
import matplotlib.collections as mcoll
import matplotlib.path as mpath


def colorline(
        x, y, z=None, cmap=plt.get_cmap('copper'), norm=plt.Normalize(0.0, 1.0),
        linewidth=3, alpha=1.0):
    """
    http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
    http://matplotlib.org/examples/pylab_examples/multicolored_line.html
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    """

    segments, z = make_segments(x, y, z)
    z = np.asarray(z)
    lc = mcoll.LineCollection(segments, array=z, cmap=cmap, norm=norm,
                              linewidth=linewidth, alpha=alpha)

    ax = plt.gca()
    ax.add_collection(lc)

    return lc


def make_segments(x, y, z, presicion_x=0.01, presicion_y=0.01):
    """
    Create list of line segments from x and y coordinates, in the correct format
    for LineCollection: an array of the form numlines x (points per line) x 2 (x
    and y) array
    """
    resample_x = [x[0]]
    resample_y = [y[0]]
    resample_z = [z[0]]
    for i, j, k in zip(x, y, z):
        if np.abs(resample_x[-1] - i) > presicion_x or np.abs(resample_y[-1] - j) > presicion_y:
            resample_x.append(i)
            resample_y.append(j)
            resample_z.append(k)
    points = np.array([resample_x, resample_y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments, resample_z
