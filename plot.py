import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import EngFormatter

from _img_dir import IMG_DIR
from util import *  # N, N_sqrt
plt.rcParams['font.family'] = 'serif'

# cmap = 'rainbow'
cmap = 'inferno'


def sci_labels(ax):
    formatter = EngFormatter(places=1, sep=u"\N{THIN SPACE}")
    ax.xaxis.set_major_formatter(formatter)


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


def scatter_multiple(y, v=None, title='', prefix='', filename=None, **kwargs):
    n_subplots = 3
    fig = plt.figure(figsize=(n_subplots * 5, 4))
    plt.suptitle(title, y=1.04, fontsize=16, fontweight='bold')
    ax = plt.subplot(1, n_subplots, 1)
    scatter(y[:, 0], v, 'Amplitude', fig=fig, **kwargs)
    sci_labels(ax)
    plt.xlabel("Space dim1 (m)")
    plt.ylabel("Space dim2 (m)")

    ax = plt.subplot(1, n_subplots, 2)
    scatter(irradiance(to_polar(y[:, 0], y[:, 1])), v, 'Irradiance', fig=fig,
            **kwargs)
    sci_labels(ax)
    plt.xlabel("Space dim1 (m)")
    plt.ylabel("Space dim2 (m)")

    # cyclic cmap: hsv, twilight
    kwargs['cmap'] = 'twilight'
    ax = plt.subplot(1, n_subplots, 3)
    scatter(y[:, 1] / np.pi, v, 'Phase', fig=fig, **kwargs)
    sci_labels(ax)
    plt.xlabel("Space dim1 (m)")
    plt.ylabel("Space dim2 (m)")

    # TODO tight layout?
    if filename is not None:
        # pdf is slow for large scatterplots
        save_fig(filename, ext='png')
    # plt.subplot(1, n_subplots, 3)
    # scatter(irradiance(to_polar(y[:, 0], y[:, 1])),
    #         v, '%s I' % prefix, lambda a, phi: a * np.sin(phi), fig=fig)
    # # scatter(y, v, '%s (a)' % prefix)
    # # scatter(y, v, '%s (a*)' % prefix, lambda a, phi: a * np.sin(phi))
    # # scatter(y, v, r'%s ($\phi$)' % prefix)
    # slice(y, v)


def entropy(H, w, title='H',  **kwargs):
    # TODO for entropy: cmap gnuplot
    n_subplots = 2
    fig = plt.figure(figsize=(n_subplots * 5, 4))
    plt.suptitle(title, y=1.02, fontsize=16, fontweight='bold')
    ax = plt.subplot(1, n_subplots,  1)
    scatter(H[:, 0], w, title='Amplitude', fig=fig, **kwargs)
    ax = plt.subplot(1, n_subplots,  2)
    scatter(H[:, 1], w, title='Phase', fig=fig, **kwargs)


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


def save_fig(filename, dir='img', ext='pdf', dpi='figure',
             transparent=True, bbox_inches='tight', interpolation='none'):
    assert(os.path.isdir(dir))
    # plt.axis('off') # this only affects the current subplot
    plt.savefig(f'{dir}/{filename}.{ext}', dpi=dpi, transparent=True,
                interpolation=interpolation, bbox_inches=bbox_inches)


if __name__ == '__main__':
    n = 100
    x = np.linspace(0, 5 * np.pi, n)
    plt.plot(x, np.sin(x))
    plt.savefig('{IMG_DIR}/tst.pdf', transparent=True)
