import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import EngFormatter

from _img_dir import IMG_DIR
from util import *  # N, N_sqrt
plt.rcParams['font.family'] = 'serif'

# cmap = 'rainbow'
cmap = 'inferno'
cyclic_cmap = 'twilight'


def scatter_multiple(x, u=None, title='', prefix='', filename=None, **kwargs):
    if 's' not in kwargs:
        n = x.shape[0]
        kwargs['s'] = max(1, 10 - n / 2.)
        print(f"n: {n}, \ts: {kwargs['s']}")

    plot_amp_phase_irradiance(_scatter_wrapper, x, u, title='',
                              filename=filename, **kwargs)


def hist_2d_multiple(x, u, title='', filename=None, bins=100, **kwargs):
    plot_amp_phase_irradiance(_hist2d_wrapper, x, u, title='',
                              filename=filename, bins=bins, **kwargs)


def hexbin_multiple(x, u, title='', filename=None,  bins=10, **kwargs):
    plot_amp_phase_irradiance(plt.hexbin, x, u, title=title,
                              filename=filename, gridsize=bins, **kwargs)


def _scatter_wrapper(x, y, z, **kwargs):
    plt.scatter(x, y, c=z, **kwargs)
    if x.shape[0] > 1:
        plt.xlim(x.min(), x.max())

    if y.shape[0] > 1:
        plt.ylim(y.min(), y.max())


def _hist2d_wrapper(x, y, z, **kwargs):
    plt.hist2d(x, y, weights=z, **kwargs)


def plot_amp_phase_irradiance(plot_func, x, v, title='', filename=None, **kwargs):
    """
    x   2d array of amp, phase
    v   3d array of spacial locations of data x
    plot_func(array1, array2, array3, pyplot_args)   e.g. plt.scatter
    """
    a, phi = x.T
    if 'cmap' not in kwargs:
        global cmap
        kwargs['cmap'] = cmap

    n_subplots = 3
    fig = plt.figure(figsize=(n_subplots * 5, 4))
    plt.suptitle(title, y=1.04, fontsize=16, fontweight='bold')

    ax = plt.subplot(131)
    plot_func(v[:, 1], v[:, 0], a, **kwargs)
    scatter_markup(ax)
    plt.title('Amplitude')

    ax = plt.subplot(133)
    plot_func(v[:, 1], v[:, 0], irradiance(to_polar(a, phi)), **kwargs)
    scatter_markup(ax)
    plt.title('Irradiance')

    # cyclic cmap: hsv, twilight
    kwargs['cmap'] = cyclic_cmap
    ax = plt.subplot(132)
    plot_func(v[:, 1], v[:, 0], phi, **kwargs)
    scatter_markup(ax)
    plt.title('Phase')

    plt.tight_layout()
    if filename is not None:
        save_fig(filename, ext='png')

    return fig


def sci_labels(ax, decimals=1, y=True, z=False):
    formatter = EngFormatter(places=decimals, sep=u"\N{THIN SPACE}")
    ax.xaxis.set_major_formatter(formatter)
    if y:
        ax.yaxis.set_major_formatter(formatter)
    if z:
        ax.zaxis.set_major_formatter(formatter)


def scatter_markup(ax):
    sci_labels(ax)
    plt.xlabel("Space dim1 (m)")
    plt.ylabel("Space dim2 (m)")
    plt.colorbar(fraction=0.052, pad=0.05)


def bitmap(x, discretize=0, filename=None, prefix='img/', scatter=0, pow=None):
    # invert to have 1 display as white pixels
    x = 1 - standardize_amplitude(x[:, 0].copy())
    if pow:
        x = x ** pow
    matrix(reshape(x, True), cmap='binary', cb=False)
    if filename is not None:
        plt.axis('off')
        plt.savefig(prefix + filename, dpi='figure',
                    transparent=True, bbox_inches='tight')


def save_fig(filename, ext='pdf', dpi='figure',
             transparent=True, bbox_inches='tight', interpolation='none'):
    assert os.path.isdir(IMG_DIR), \
        '_img_dir.py/IMG_DIR is must be setup correctly'
    # plt.axis('off') # this only affects the current subplot
    plt.savefig(f'{IMG_DIR}/{filename}.{ext}', dpi=dpi, transparent=True,
                interpolation=interpolation, bbox_inches=bbox_inches)


###############################################################################
# Deprecated
###############################################################################

def vectors(X, labels=('x', 'y', 'z'), title='', **kwargs):
    # X : list(np.ndarray)
    data = ['a', r'$\phi$', 'I']
    n_subplots = X[0].shape[1] + 1
    fig = plt.figure(figsize=(n_subplots * 5, 4))
    plt.suptitle(title, y=1.02, fontsize=16, fontweight='bold')
    for i in range(n_subplots):
        ax = plt.subplot(1, n_subplots, i + 1)
        plt.title(data[i])
        for j, x in enumerate(X):
            if len(X) == 1:
                marker = '-'
            else:
                marker = ['-.', ':', '--'][j % 3]

            if i < n_subplots - 1:
                plt.plot(X[j][:, i], marker, label=labels[j], **kwargs)
            else:
                plt.plot(irradiance(to_polar(*split_wave_vector(X[j]))),
                         marker, label=labels[j], **kwargs)
        plt.legend()
        plt.margins(0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    return fig


def matrix(x, title='', cb=True, fig=None, log=False, **kwargs):
    if 'cmap' not in kwargs:
        global cmap
        kwargs['cmap'] = cmap

    if fig is None:
        plt.figure()

    vmin = 0
    if log:
        x = semilog(x)
        vmin = None

    if len(x.shape) > 1:
        plt.imshow(x, vmin=vmin, origin='lower', **kwargs)
    else:
        plt.imshow(x.reshape(x.size, 1), vmin=vmin, origin='lower', **kwargs)
    # plt.colorbar(fraction=0.046, pad=0.04)
    if cb:
        plt.colorbar(fraction=0.035, pad=0.04)
    plt.title(title)
    if fig is None:
        plt.tight_layout()

    return fig


def matrix_multiple(y, title='y', prefix='', m=2, HD=0, filename=None, **kwargs):
    a, phi = split_wave_vector(y, HD)

    # data = ['a', r'$\phi$', 'I']
    n_subplots = m * 3
    fig = plt.figure(figsize=(n_subplots // m * 5, m * 5))
    plt.suptitle(title, y=1.04, fontsize=16, fontweight='bold')
    plt.subplot(m, n_subplots // m, 1)
    matrix(reshape(y[:, 0], HD), '%s Amplitude' % prefix, fig=fig, **kwargs)
    plt.xlabel("Space dim1 (m)")
    plt.ylabel("Space dim2 (m)")
    plt.subplot(m, n_subplots // m, 2)
    matrix(reshape(y[:, 0], HD), '%s Log Amplitude' %
           prefix, fig=fig, log=True, **kwargs)
    plt.subplot(m, n_subplots // m, 3)
    # cyclic cmap: hsv, twilight
    matrix(reshape(y[:, 1], HD) / np.pi, '%s Phase' %
           prefix, fig=fig, cmap='twilight', **kwargs)
    if m >= 2:
        plt.subplot(m, n_subplots // m, 4)
        matrix(irradiance(to_polar(a, phi)), '%s I (norm)' %
               prefix, fig=fig, **kwargs)
        plt.subplot(m, n_subplots // m, 5)
        # matrix(a * np.cos(phi), '%s (a*)' % prefix)
        matrix(irradiance(to_polar(a, phi), normalize=0), '%s I' %
               prefix, fig=fig, **kwargs)
        plt.subplot(m, n_subplots // m, 6)
        matrix(a * np.cos(phi * 2 * np.pi), r'%s A cos $\phi$' %
               prefix, fig=fig, **kwargs)
        # matrix(a, '%s A cos phi' % prefix, fig=fig, **kwargs)
        # matrix(np.cos(phi * np.pi), '%s A cos phi' % prefix, fig=fig, **kwargs)

    plt.tight_layout()
    # if filename is not None:
    #     plt.savefig(filename + '.pdf', transparent=True)
    if filename is not None:
        save_fig(filename, ext='pdf')

    if m >= 2:
        if DIMS > 2:
            slice(y, HD=HD)
            # if filename is not None:
            #     plt.savefig(filename + '-slice.pdf', transparent=True)
            if filename is not None:
                save_fig(f'{filename}-slice', ext='pdf')


def slice(y, v=None, HD=0):
    a, phi = split_wave_vector(y, HD)
    plt.figure(figsize=(6, 3))

    plt.subplot(121)
    b = np.mean(irradiance(to_polar(a, phi)), axis=1)
    plt.plot(b)
    plt.title('slice 1')

    plt.subplot(122)
    b = np.mean(irradiance(to_polar(a, phi)), axis=0)
    plt.plot(b)
    plt.title('slice 2')

    plt.tight_layout()

    if v is not None:
        plt.figure(figsize=(6, 3))
        plt.subplot(121)
        b = irradiance(to_polar(a, phi))
        plt.scatter(v[:, 0], b, s=2, alpha=0.8, color='0')
        plt.subplot(122)
        plt.scatter(v[:, 1], b, s=2, alpha=0.8, color='0')
        plt.tight_layout()


def scatter(x, w, title='', color_func=lambda a, phi: a, log=False, s=10,
            alpha=0.9, fig=None, **kwargs):
    # x : shape (N,)
    if 'cmap' not in kwargs:
        global cmap
        kwargs['cmap'] = cmap

    if fig is None:
        fig = plt.figure()
    if log:
        x = semilog(x)
    plt.scatter(w[:, 1], w[:, 0], c=x, s=s, alpha=alpha,  **kwargs)
    plt.xlim(w[:, 1].min(), w[:, 1].max())
    plt.ylim(w[:, 0].min(), w[:, 0].max())
    plt.colorbar(fraction=0.052, pad=0.05)
    # plt.colorbar(fraction=0.046, pad=0.04)
    plt.title(title)
    plt.tight_layout()
    return fig


def entropy(H, w, title='H',  **kwargs):
    # TODO for entropy: cmap gnuplot
    n_subplots = 2
    fig = plt.figure(figsize=(n_subplots * 5, 4))
    plt.suptitle(title, y=1.02, fontsize=16, fontweight='bold')
    ax = plt.subplot(1, n_subplots,  1)
    scatter(H[:, 0], w, title='Amplitude', fig=fig, **kwargs)
    # scatter_markup(ax)
    ax = plt.subplot(1, n_subplots,  2)
    scatter(H[:, 1], w, title='Phase', fig=fig, **kwargs)
    # scatter_markup(ax)
