import numpy as np
import itertools
import matplotlib.pyplot as plt
import matplotlib.animation as animation
# local
from _img_dir import IMG_DIR
import plot
import util
from util import DIMS


def single(X, U, matrix_func, color_func,
           title='', filename=None, fps=20, upsampling=10, repetitions=1,
           offsets=[], plot_kwargs={}, **matrix_func_kwargs):
    # sample_rate = number of samples per wave cycle
    # T = 1/f
    dt = 1e3 / fps * 10

    assert(len(X) == len(U))
    if 'cmap' not in matrix_func_kwargs:
        global cmap
        matrix_func_kwargs['cmap'] = cmap

    cmap = matrix_func_kwargs['cmap']
    fig, ax, im = plot_matrix(
        matrix_func(X[0], U[0], color_func, **matrix_func_kwargs),
        cmap=cmap, **plot_kwargs)

    # force aspect ratio
    ax.set_aspect(1.0 / ax.get_data_ratio() / (1920 / 1080))

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
                **matrix_func_kwargs))

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
        # , bitrate=256
        ani.save(f'{IMG_DIR}/{filename}.mp4',
                 fps=fps, extra_args=['-vcodec', 'libx264'])

    # show in ipython notebook (incompatible with frames arg in FuncAnimation)
    # plt.close()
    # return HTML(ani.to_html5_video())


def multiple(X: list, U: list, prefix='',
             matrix_func=None, **kwargs):
    # TODO simplify interface
    titles = ['Amplitude', 'Phase', 'Irradiance']
    color_funcs = [lambda x: x[:, 0],
                   lambda x: x[:, 1],
                   lambda x: util.irradiance(util.to_polar(*x.T))]
    cmaps = [plot.cmap, plot.cyclic_cmap, plot.cmap]
    if matrix_func is None:
        matrix_func = histogram

    for i in range(2):
        Xi = [x[:, i] for x in X]
        single(Xi, U, matrix_func, color_func=color_funcs[i],
               title=titles[i], filename=f'{prefix}-{titles[i]}',
               cmap=cmaps[i],  **kwargs)


def multiple_hd(ratio, *args, **kwargs):
    multiple(*args, matrix_func=plot.hist_2d_hd,
             plot_kwargs={'markup': False, 'ratio': ratio},
             **kwargs)

###############################################################################
# Helper functions
###############################################################################


def plot_matrix(h: np.ndarray, title='z', fig=None, markup=True, ratio=None,
                **kwargs):
    if fig is None:
        fig = plt.figure(figsize=(5, 4))

    print('plot_matrix', h.shape, markup, ratio)

    ax = plt.gca()
    im = plt.imshow(h, origin='lower', **kwargs)
    if markup:
        plot.markup(ax)
        # TODO print z-offset
    else:
        # fullscreen mode
        plt.axis('off')
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                            hspace=0, wspace=0)

    if ratio is not None:
        # force aspect ratio
        ax.set_aspect(1.0 / ax.get_data_ratio() / ratio)

    return fig, ax, im


def histogram(x, u, color_func, **kwargs) -> np.ndarray:
    # matrix func
    # Returns a histogram matrix
    colors = color_func(x)
    assert(u[:, 0].size == colors.size)
    return plt.hist2d(u[:, 1], u[:, 0], weights=colors, **kwargs)[0]
