import numpy as np
import itertools
import matplotlib.pyplot as plt
import matplotlib.animation as animation
# local
from _img_dir import IMG_DIR
import plot
import util
from util import DIMS


def single(X, U, plot_func, matrix_func, color_func, title='', filename=None,
           fps=1, repetitions=2,  **matrix_func_args):
    # sample_rate = number of samples per wave cycle
    # T = 1/f
    dt = 1e3 / fps * 10

    assert(len(X) == len(U))
    if 'cmap' not in matrix_func_args:
        global cmap
        matrix_func_args['cmap'] = cmap

    cmap = matrix_func_args['cmap']
    fig, im = plot_func(matrix_func(X[0], U[0], color_func, **matrix_func_args),
                        cmap=cmap)

    # FuncAnimation module requires a subfunction
    def update(data, im):
        im.set_array(matrix_func(*data, color_func, **matrix_func_args))
        return im,

    # note that zip can only be iterated once
    data_per_frame = itertools.chain.from_iterable(
        (zip(X, U) for _ in range(repetitions)))
    ani = animation.FuncAnimation(fig, update, data_per_frame, fargs=(im,),
                                  interval=dt, blit=True)
    if filename is not None:
        # Save a copy of the object
        # (saving may affect the video playback in jupyter)
        ani.save(f'{IMG_DIR}/{filename}.mp4',
                 fps=fps, extra_args=['-vcodec', 'libx264'])

    # show in ipython notebook (incompatible with frames arg in FuncAnimation)
    # plt.close()
    # return HTML(ani.to_html5_video())


def multiple(X: list, U: list, prefix='', bins=100):
    titles = ['Amplitude', 'Phase', 'Irradiance']
    color_funcs = [lambda x: x[:, 0],
                   lambda x: x[:, 1],
                   lambda x: util.irradiance(util.to_polar(*x.T))]
    cmaps = [plot.cmap, plot.cyclic_cmap, plot.cmap]
    for i in range(3):
        single(X, U, plot_func=heatmap, matrix_func=histogram,
               color_func=color_funcs[i], title=titles[i],
               filename=f'{prefix}-{titles[i]}', bins=bins, cmap=cmaps[i])

###############################################################################
# Helper functions
###############################################################################


def heatmap(h: np.ndarray, title='z', **kwargs):
    fig = plt.figure(figsize=(5, 4))
    ax = plt.gca()
    im = plt.imshow(h, origin='lower', **kwargs)
    plot.scatter_markup(ax)
    # TODO print z-offset
    return fig, im


def histogram(x, u, color_func, **kwargs) -> np.ndarray:
    # Returns a histogram matrix
    colors = color_func(x)
    assert(u[:, 0].size == colors.size)
    return plt.hist2d(u[:, 1], u[:, 0], weights=colors, **kwargs)[0]
