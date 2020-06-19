import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import EngFormatter, LogLocator, LogFormatter, LogFormatterSciNotation

from _img_dir import IMG_DIR
import util
from util import *  # N, N_sqrt
plt.rcParams['font.family'] = 'serif'

# cmap = 'rainbow'
cmap = 'inferno'
cyclic_cmap = 'twilight'


def hist_2d_hd(phasor, u, title='', filename=None,  ybins=10, ratio=1,
               cmap='gray', bin_threshold=0.1, verbose=1, **kwargs):
    Nx, Ny = util.solve_xy_is_a(u.shape[0], ratio)
    ybins = int(util.find_nearest_denominator(Ny, ybins, bin_threshold))
    if Nx == Ny:
        # ratio is close to 1.0
        xbins = ybins
        assert xbins <= Nx
    else:
        # derive xbins from updated ybins to preserve "squareness" of pixels
        xbins = min(Nx, round(ybins * ratio))
        xbins = util.find_nearest_denominator(Nx, xbins, bin_threshold)

    if verbose:
        print('bins:',
              f'\tx: {xbins} (~{Nx / xbins:0.1f} per bin)\n',
              f'\ty: {ybins} (~{Ny / ybins:0.1f} per bin)')

    assert xbins <= Nx
    assert ybins <= Ny
    x = u[:, 0]
    y = u[:, 1]
    # bins = (np.linspace(x.min(), x.max(), xbins + 1)[1:-1],
    #         np.linspace(y.min(), y.max(), ybins + 1)[1:-1])
    bins = util.regular_bins([x.min(), y.min()],
                             [x.max(), y.max()],
                             [xbins, ybins])
    # bins = (xbins, ybins)
    h = 4
    w = round(h * ratio)
    # print(w, h, w / h)
    items = ['amp', 'phase']
    n_items = (phasor.shape + (1,))[1]
    for i, k in enumerate(items[:n_items]):
        plt.figure(figsize=(w, h))
        # ax = plt.gca()
        ax = plt.subplot()
        color = phasor if len(phasor.shape) == 1 else phasor[:, i]
        # TODO use imshow for nonrand?
        matrix = _hist2d_wrapper(x, y, color, bins=bins, cmap=cmap, **kwargs)
        plt.axis('off')

        # force aspect ratio
        ax.set_aspect(1.0 / ax.get_data_ratio() / ratio)

        # plt.tight_layout()
        if filename is not None:
            save_fig(f'{filename}_{k}', ext='png', pad_inches=0)

    return matrix


def scatter_multiple(x, u=None, title='', subtitle='', filename=None, **kwargs):
    if 's' not in kwargs:
        # set point size
        n = x.shape[0]
        kwargs['s'] = max(1, 10 - n / 2.)

    amp_phase_irradiance(_scatter_wrapper, x, u,
                         title=title, subtitle=subtitle,
                         filename=filename, **kwargs)


def hist_2d_multiple(phasor, pos, title='', subtitle='', filename=None,
                     ybins=100, ratio=1., bin_threshold=0.1, **kwargs):
    """
    Plot 2d histogram

    Params
    ------
    phasor : array of (amp, phase) (for each datapoint)
        (e.g. x,y,z as used in other funcs)
    pos : flattened 3d array with 3D positions for each datapoint
        (e.g. u,v,w of corresponding positions)

    xbins, ybins : horizontal and vertical bins (respectively) and relate to
    the plot/image; not to the phasor.
    """
    assert ratio != 0
    # TODO fix bins
    Nx, Ny = util.solve_xy_is_a(pos.shape[0], ratio)
    ybins = util.find_nearest_denominator(Ny, ybins, bin_threshold)
    if Nx == Ny:
        # ratio is close to 1.0
        xbins = ybins
        assert xbins <= Nx
    else:
        # derive xbins from updated ybins to preserve "squareness" of pixels
        xbins = min(Nx, round(ybins * ratio))
        xbins = util.find_nearest_denominator(Nx, xbins, bin_threshold)

    assert xbins <= Nx
    assert ybins <= Ny
    x = pos[:, 0]
    y = pos[:, 1]
    bins = util.regular_bins([x.min(), y.min()],
                             [x.max(), y.max()],
                             [xbins, ybins])

    amp_phase_irradiance(_hist2d_wrapper, phasor, pos,
                         title=title, subtitle=subtitle,
                         filename=filename, bins=bins, ratio=ratio,
                         density3=False, **kwargs)

    # if not randomized:
    # amp_phase_irradiance(_imshow_wrapper, phasor, pos,
    #                      title=title, subtitle=subtitle,
    #                      filename=filename, ratio=ratio,
    #                      **kwargs)


def _scatter_wrapper(x, y, z, **kwargs):
    rel_margin = 0.05
    plt.scatter(x, y, c=z, **kwargs)
    plt.axhline(0, color='0', ls='--', lw=1, alpha=0.4)
    plt.axvline(0, color='0', ls='--', lw=1, alpha=0.4)
    if x.shape[0] == 1:
        return

    width = abs(x.max() - x.min())
    height = abs(y.max() - y.min())
    max_width = max(width, height)
    args = [(x, plt.xlim), (y, plt.ylim)]
    if width < height:
        args.reverse()

    for s, lim_func in args:
        a, b = s.min(), s.max()
        width = abs(b - a)  # idem for height
        if a == b:
            return

        # aspect ratio is 1:1, set corresponding limits
        # assume width >= height
        if width < max_width:
            a -= (max_width - width) / 2
            b += (max_width - width) / 2

        if x.shape[0] < 20:
            # for small datasets
            margin = max_width * rel_margin
            a -= margin
            b += margin
            # # origin 0,0 should be in included
            # TODO this invalidates max_width
            # a = min(0, a)
            # b = max(0, b)

        lim_func(a, b)


def _hist2d_wrapper(x, y, z, density=True, **kwargs):
    return plt.hist2d(x, y, weights=z, density=density, **kwargs)[0]


def _imshow_wrapper(x, _y, _z, ratio=1., **kwargs):
    # TODO use imshow for nonrand planes to avoid unnecessary computation
    # TODO fix return type to be fully compatible with _hist2d_wrapper?
    hd = ratio > 1.
    # return plt.imshow(x, y, weights=z, density=density, **kwargs)
    # vmin, vmax
    plt.imshow(reshape(z[:, 0], hd), origin='lower', aspect='auto', **kwargs)


def amp_phase_irradiance(plot_func, x, v, title='', subtitle='', filename=None,
                         ratio=1., density3=None, large=True, **kwargs):
    """ Triple plot of amplitude, phase, irradiance

    Params
    ------
    x : array of shape (N, 2)
        representing amplitude, phase of N datapoints
    v : array of shape (N, 3)
        the corresponding 3d positions of x
    plot_func : func that takes args (array1, array2, array3, pyplot_args)
        e.g. plt.scatter
    """
    a, phi = x.T
    if 'cmap' not in kwargs:
        global cmap
        kwargs['cmap'] = cmap

    n_subplots = 3
    horizontal = ratio < 1.1  # distribute subplots horizontally
    # h = 15, w = 4
    if horizontal:
        if large:
            w = n_subplots * ratio * 7
            h = 6
        else:
            w = n_subplots * ratio * 5
            h = 4
    else:
        h = n_subplots * ratio * 2 + 1
        w = 8

    fig = plt.figure(figsize=(round(w), h))
    plt.suptitle(title, y=1.02, fontsize=16, fontweight='bold')

    if horizontal:
        ax = plt.subplot(131)
    else:
        ax = plt.subplot(311)

    v1 = v[:, 0]
    v2 = v[:, 1]
    plot_func(v1, v2, a, **kwargs)
    markup(ax, unit='m')
    plt.title('Amplitude', fontsize=16)

    # change subplot order to allow cmap to be changed later
    # ax = plt.subplot(133)
    if horizontal:
        ax = plt.subplot(133)
    else:
        ax = plt.subplot(313)

    # |E|^2 == irradiance(E)
    log_irradiance = np.log(np.clip(a ** 2, 1e-9, None))
    if density3 is not None:
        # hack to allow optional 3rd param for histogram plot func
        kwargs['density'] = density3

    plot_func(v1, v2, standardize(log_irradiance), **kwargs)
    if density3 is not None:
        del kwargs['density']

    markup(ax, unit='m')
    plt.title('Log Irradiance', fontsize=16)

    # ax = plt.subplot(132)

    if horizontal:
        ax = plt.subplot(132)
    else:
        ax = plt.subplot(312)

    # cyclic cmap: hsv, twilight
    kwargs['cmap'] = cyclic_cmap
    plot_func(v1, v2, phi, **kwargs)
    markup(ax, unit='m')
    plt.title('Phase', fontsize=16)

    # if horizontal:
    plt.text(0.5, 1.15, subtitle, {'fontsize': 12},
             horizontalalignment='center', verticalalignment='center',
             transform=ax.transAxes)

    if not horizontal:
        print('TODO')
        # plt.text(0., 0.0, subtitle, {'fontsize': 14})

    # plt.text(0., 0.0015, 'abc', {'fontsize': 14})
    try:
        plt.tight_layout()

        # add custom subtitle after tightening layout
        # plt.text(0.5, 0, 'sec', {'fontsize': 14})
        # plt.text(0.5, 0, 'subtitle\nab\nasb\nasldkfjasldjfskdf asdfasdf',
        #          {'fontsize': 18})
        # plt.text(0.5, 0.5, 'subtitle\nab\nasb\nasldkfjasldjfskdf asdfasdf', {
        #          'fontsize': 18})
        if filename is not None:
            save_fig(filename, ext='png')

    except ValueError as e:
        print(e)
    return fig


def sci_labels(ax, decimals=1, y=True, z=False, unit='', rotation=30):
    formatter = EngFormatter(places=decimals, sep=u"\N{THIN SPACE}", unit=unit)
    ax.xaxis.set_major_formatter(formatter)
    plt.xticks(rotation=rotation)
    if y:
        ax.yaxis.set_major_formatter(formatter)
    if z:
        # 3D plot
        ax.zaxis.set_major_formatter(formatter)


def markup(ax, unit=''):
    sci_labels(ax, unit=unit)
    plt.xlabel("Space dimension 1")
    plt.ylabel("Space dimension 2")
    # plt.colorbar(fraction=0.052, pad=0.05,
    #              ticks=LogLocator(subs='all'), format=LogFormatterSciNotation())
    plt.colorbar(fraction=0.052, pad=0.05)


def format_title(major, minor, info: dict = None):
    """ e.g. `x_0 (distance: 10cm)` """
    title = f"{major}$_{{{minor}}}$"
    if info is not None:
        sub = ', '.join([f"{k}: ${round(v, 2)}$" for k, v in info.items()])
        title += f" ({sub})"
    return title


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
             transparent=True, bbox_inches='tight', interpolation='none',
             **kwargs):
    assert os.path.isdir(IMG_DIR), \
        '_img_dir.py/IMG_DIR is must be setup correctly'
    # plt.axis('off') # this only affects the current subplot
    plt.savefig(f'{IMG_DIR}/{filename}.{ext}', dpi=dpi, transparent=True,
                interpolation=interpolation, bbox_inches=bbox_inches,
                **kwargs)
    plt.close()


###############################################################################
# Deprecated
###############################################################################

def vectors(X, labels=('x', 'y', 'z'), title='', **kwargs):
    # X : list(np.ndarray)
    data = ['a', r'$\phi$', 'I']
    n_subplots = X[0].shape[1] + 1
    fig = plt.figure(figsize=(n_subplots * 5, 4))
    plt.suptitle(title, y=1.04, fontsize=16, fontweight='bold')
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
        # plt.subplot(m, n_subplots // m, 6)
        # matrix(a * np.cos(phi * 2 * np.pi), r'%s A cos $\phi$' %
        #        prefix, fig=fig, **kwargs)
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


if __name__ == '__main__':
    sequence_dir = util.get_arg('--sequence_dir', '', parse_func=str)
    dir = '../tmp'
    fn = 'out.zip'
    size = os.path.getsize(os.path.join(dir, fn))
    print(f'Input file size: {size * 1e-6:0.5f} MB')
    if size > 1e6:
        print(f'Warning, file too large: {size*1e-6:0.4f} MB')

    params, data = util.parse_file(dir, fn, 'out')
    # print('uu', data['u'])

    N = data['y'][0].shape[0]
    ratio = 1920. / 1080. if params['y'][0]['hd'] else 1.
    Nx, Ny = util.solve_xy_is_a(N, ratio)
    Nxy = Nx * Ny
    for k in 'yv':
        data[k] = data[k][:Nxy]

    print({'N': N, 'Nx': Nx, 'Ny': Ny, 'eq': Nx * Ny == N})
    N_sqrt = int(np.sqrt(N))
    print(f'N sqrt (y): {N_sqrt}')
    # max_n_plots = 2
    # m = len(data['y'])
    # args = (m,) if m <= max_n_plots else (
    #     0, m, np.ceil(m / max_n_plots).astype(int))
    # for i in range(*args):
    #     surf_multiple(data['y'][i], data['v'][i], Nx, Ny f'$y_{i} $')

    m = len(data['y'])
    args1 = (m,) if m <= 2 else (0, m, np.ceil(m / 2).astype(int))
    n_z_per_y = len(data['z']) // len(data['y'])
    m = n_z_per_y
    args2 = (m,) if m <= 1 else (0, m, np.ceil(m / 2).astype(int))
    for major in range(*args1):
        for minor in range(*args2):
            i = major * n_z_per_y + minor
            offset = params['z'][i]['z_offset']
            title = f"$z_{{{major}, {minor}}}$ " + \
                f"(offset: {round(offset, 2)} m)"
            print(title)
            prefix = f'$z_{{{major},{minor}}}$ '
            # matrix(data['z'][i], data['w'][i], N_sqrt, N_sqrt, prefix)
            z = data['z'][i]
            if params['z'][i]['randomized']:
                print('Warning, plotting random points without proper discretization')
            matrix(reshape(z[:, 0], params['x'][i]['hd']), f'z_{i} Amplitude')
            plt.show()
