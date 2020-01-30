import numpy as np
import matplotlib.pyplot as plt

from util import *  # N, N_sqrt

# cmap = 'rainbow'
cmap = 'inferno'


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


def matrix(x, title='', fig=None, **kwargs):
    if 'cmap' not in kwargs:
        global cmap
        kwargs['cmap'] = cmap

    if fig is None:
        plt.figure()
    if len(x.shape) > 1:
        plt.imshow(x, vmin=0, origin='lower', **kwargs)
    else:
        plt.imshow(x.reshape(x.size, 1), vmin=0, origin='lower', **kwargs)
    # plt.colorbar(fraction=0.046, pad=0.04)
    plt.colorbar(fraction=0.035, pad=0.04)
    plt.title(title)
    if fig is None:
        plt.tight_layout()


def matrix_multiple(y, title='y', prefix='', m=2, HD=0):
    a, phi = split_wave_vector(y, HD)

    # data = ['a', r'$\phi$', 'I']
    n_subplots = m * 3
    fig = plt.figure(figsize=(n_subplots // m * 4, m * 4))
    plt.suptitle(title, y=1.02, fontsize=16, fontweight='bold')
    plt.subplot(m, n_subplots // m, 1)
    matrix(reshape(y[:, 0], HD), '%s Amplitude' % prefix, fig=fig)
    plt.subplot(m, n_subplots // m, 2)
    # cyclic cmap: hsv, twilight
    matrix(reshape(y[:, 1], HD) / np.pi, r'%s $\phi$' %
           prefix, fig=fig, cmap='twilight')
    if m >= 2:
        plt.subplot(m, n_subplots // m, 4)
        matrix(irradiance(to_polar(a, phi)), '%s I (norm)' % prefix, fig=fig)
        plt.subplot(m, n_subplots // m, 5)
        # matrix(a * np.cos(phi), '%s (a*)' % prefix)
        matrix(irradiance(to_polar(a, phi), normalize=0), '%s I' %
               prefix, fig=fig)
        plt.subplot(m, n_subplots // m, 6)
        matrix(a * np.cos(phi * 2 * np.pi), '%s A cos phi' % prefix, fig=fig)
        # matrix(a, '%s A cos phi' % prefix, fig=fig)
        # matrix(np.cos(phi * np.pi), '%s A cos phi' % prefix, fig=fig)
    plt.tight_layout()
    if m >= 2:
        if DIMS > 2:
            slice(y, HD=HD)


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


def scatter(x, w, title='', color_func=lambda a, phi: a, s=10, alpha=0.9,
            fig=None, **kwargs):
    # x : shape (N,)
    if 'cmap' not in kwargs:
        global cmap
        kwargs['cmap'] = cmap

    if fig is None:
        fig = plt.figure()
    plt.scatter(w[:, 1], w[:, 0], c=x, s=s,
                alpha=alpha,  **kwargs)
    plt.xlim(w[:, 1].min(), w[:, 1].max())
    plt.ylim(w[:, 0].min(), w[:, 0].max())
    plt.colorbar(fraction=0.052, pad=0.05)
    # plt.colorbar(fraction=0.046, pad=0.04)
    plt.title(title)
    plt.tight_layout()
    return fig


def scatter_multiple(y, v=None, title='', prefix='', **kwargs):
    n_subplots = 3
    fig = plt.figure(figsize=(n_subplots * 4, 3))
    plt.suptitle(title, y=1.02, fontsize=16, fontweight='bold')
    plt.subplot(1, n_subplots, 1)
    scatter(y[:, 0], v, '%s Amplitude' % prefix, fig=fig, **kwargs)
    plt.subplot(1, n_subplots, 2)
    scatter(irradiance(to_polar(y[:, 0], y[:, 1])),
            v, r'%s I' % prefix, fig=fig, **kwargs)
    plt.subplot(1, n_subplots, 3)
    # cyclic cmap: hsv, twilight
    kwargs['cmap'] = 'twilight'
    scatter(y[:, 1] / np.pi, v, r'%s $\phi$' % prefix, fig=fig, **kwargs)
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
    scatter(H[:, 1], w, title=r'$\phi$', fig=fig, **kwargs)
