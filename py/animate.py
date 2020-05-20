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
           fps=20, upsampling=10, repetitions=1, offsets=[],  **matrix_func_args):
    # sample_rate = number of samples per wave cycle
    # T = 1/f
    dt = 1e3 / fps * 10

    assert(len(X) == len(U))
    if 'cmap' not in matrix_func_args:
        global cmap
        matrix_func_args['cmap'] = cmap

    cmap = matrix_func_args['cmap']
    fig, ax, im = plot_func(
        matrix_func(X[0], U[0], color_func, **matrix_func_args),
        cmap=cmap)

    if offsets:
        ax.set_title(f'z (offset: {offsets[0]} m)')

    # FuncAnimation module requires a subfunction
    def update(frame_idx, data, ax, im, offsets):
        i = frame_idx // upsampling
        if frame_idx - i * upsampling == 0:
            if len(offsets) > 1:
                ax.set_title(f'z (offset: {offsets[i]} m)')
            im.set_array(matrix_func(
                X[i], U[i],
                color_func,
                **matrix_func_args))

        return im,

    # note that the return value of zip() can only be iterated once
    data_per_frame = itertools.chain.from_iterable(
        itertools.cycle((zip(X, U) for _ in range(repetitions))))
    ani = animation.FuncAnimation(fig, update, frames=upsampling * len(X),
                                  fargs=(data_per_frame, ax, im, offsets),
                                  interval=dt, blit=True)
    if filename is not None:
        # Save a copy of the object
        # (saving may affect the video playback in jupyter)
        ani.save(f'{IMG_DIR}/{filename}.mp4',
                 fps=fps, extra_args=['-vcodec', 'libx264'])

    # show in ipython notebook (incompatible with frames arg in FuncAnimation)
    # plt.close()
    # return HTML(ani.to_html5_video())


def multiple(X: list, U: list, prefix='', bins=100, offsets=[]):
    titles = ['Amplitude', 'Phase', 'Irradiance']
    color_funcs = [lambda x: x[:, 0],
                   lambda x: x[:, 1],
                   lambda x: util.irradiance(util.to_polar(*x.T))]
    cmaps = [plot.cmap, plot.cyclic_cmap, plot.cmap]
    for i in range(2):
        single(X, U, plot_func=heatmap, matrix_func=histogram,
               color_func=color_funcs[i], title=titles[i],
               filename=f'{prefix}-{titles[i]}', bins=bins, cmap=cmaps[i],
               offsets=offsets)

###############################################################################
# Helper functions
###############################################################################


def heatmap(h: np.ndarray, title='z', **kwargs):
    fig = plt.figure(figsize=(5, 4))
    ax = plt.gca()
    im = plt.imshow(h, origin='lower', **kwargs)
    plot.scatter_markup(ax)
    # TODO print z-offset
    return fig, ax, im


def histogram(x, u, color_func, **kwargs) -> np.ndarray:
    # Returns a histogram matrix
    colors = color_func(x)
    assert(u[:, 0].size == colors.size)
    return plt.hist2d(u[:, 1], u[:, 0], weights=colors, **kwargs)[0]
