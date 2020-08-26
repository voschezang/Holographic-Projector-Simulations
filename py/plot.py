import numpy as np
import pandas as pd
import os
import scipy.stats
import scipy.signal
import matplotlib.pyplot as plt
import matplotlib.ticker as tck

from _img_dir import IMG_DIR
import util
from util import *  # N, N_sqrt
plt.rcParams['font.family'] = 'serif'

# cmap = 'rainbow'
cmap = 'inferno'
cyclic_cmap = 'twilight'

COLORS = plt.rcParams['axes.prop_cycle'].by_key()['color']
HATCHES = ('-', '+', 'x', '\\', '*', 'o', 'O', '.')
# dot-dash syntax (0, (width_i, space_i, width_j, space_j, ..))
dot1, dot2, dot3 = (1, 1), (1, 2), (1, 3)
dash1, dash2, dash3 = (2, 1), (3, 1), (4, 1)
LINESTYLES = ['-', '--', '-.', ':',
              (0,  dot1 + dot1 + dash3), (0, dash3 + dash2 + dash1 + dot1),
              (0, dot1 + dot3)]


def hist_2d_hd(phasor, pos, title='', filename=None,  ybins=10, ratio=1,
               cmap='gray', bin_threshold=0.1, bin_options={},  verbose=1,
               **kwargs):
    # Return a histogram plot without plot markup or labels
    x = pos[:, 0]
    y = pos[:, 1]
    # TODO derived minmax is incorrect: e.g. for nonrand the first point is at the boundary, for rand boundary is nondeterministic
    bins = util.gen_bin_edges(x, y, ratio, ybins, bin_threshold, bin_options)
    n_items = (phasor.shape + (1,))[1]
    items = ['amp', 'phase'] if n_items > 1 else ['']
    for i, k in enumerate(items[:n_items]):
        # plt.figure(figsize=(w, h))
        # ax = plt.subplot()
        color = phasor if len(phasor.shape) == 1 else phasor[:, i]
        # TODO use imshow for nonrand?

        # soft round required because hist2d is lossy
        # e.g. a constant input can become noisy
        matrix = np.histogram2d(x, y, weights=util.soft_round(color, verbose=1),
                                bins=bins, density=True, **kwargs)[0]
        matrix = util.soft_round(matrix, verbose=2)
        # matrix = plt.hist2d(x, y, weights=color, bins=bins, cmap=cmap, density=True, **kwargs)
        # plt.axis('off')
        # # force aspect ratio
        # ax.set_aspect(1.0 / ax.get_data_ratio() / ratio)
        # # save_fig(f'{filename}_{k}', ext='png', pad_inches=0)
        # plt.show()
        # plt.close()
        if filename is not None:
            plt.imsave(f'{IMG_DIR}/{filename}_{k}.png', matrix.T,
                       origin='lower', cmap=cmap)

    return matrix


def scatter_multiple(x, u=None, title='', subtitle='', filename=None, **kwargs):
    if 's' not in kwargs:
        # set point size
        n = x.shape[0]
        kwargs['s'] = max(1, 10 - n / 2.)

    amp_phase_irradiance(_scatter_wrapper, u[:, 0], u[:, 1], x,
                         title=title, subtitle=subtitle,
                         filename=filename, **kwargs)


def hist_2d_multiple(phasor, pos, title='', subtitle='', filename=None,
                     ybins=100, ratio=1., bin_threshold=0.1, bin_options={},
                     verbose=0, **kwargs):
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
    bins = util.gen_bin_edges(pos[:, 0], pos[:, 1], ratio, ybins,
                              bin_threshold, bin_options)
    if verbose:
        for i in [0, 1]:
            print('hist_2d_multiple', bin_options)
            print(f'bins {i}: [{bins[i].min()} ; {bins[i].max()}], n:',
                  bins[i].size)
            print(f'pos: [{pos[:,i].min()} ; {pos[:,i].max()}]')

    # TODO for a constant input the histogram won't be constant
    # use imshow(soft_round(hist(..)))
    amp_phase_irradiance(_hist2d_wrapper, pos[:, 0], pos[:, 1], phasor,
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


def _hist2d_wrapper(x, y, z, density=True, bins=10, range=None, **kwargs):
    # create tmp figure
    # fig = plt.figure()
    # hist = plt.hist2d(x, y, weights=z, density=density, bins=bins)[0]
    # plt.close(fig)
    hist = np.histogram2d(x, y, weights=z, density=density, bins=bins)[0]
    if range is not None:
        hist = util.standardize(hist) * (range[1] - range[0]) + range[0]
    # supply bins as positions s.t. the axis range equals the bins range
    _imshow_wrapper(bins[0], bins[1], hist, **kwargs)


def _imshow_wrapper(x, y, color, ratio=1., **kwargs):
    # TODO use imshow for nonrand planes to avoid unnecessary computation
    # TODO fix return type to be fully compatible with _hist2d_wrapper?
    # hd = ratio > 1.
    # TODO reshape according to non-hd ratios
    # return plt.imshow(x, y, weights=z, density=density, **kwargs)
    # vmin, vmax
    # plt.imshow(reshape(z[:, 0], hd), origin='lower', aspect='auto', **kwargs)
    plt.imshow(util.soft_round(color, verbose=3).T, origin='lower', aspect=ratio,
               extent=(x.min(), x.max(), y.min(), y.max()),
               vmin=color.min(), vmax=color.max(),
               **kwargs)


def amp_phase_irradiance(plot_func, x, y, phasor, title='', subtitle='', filename=None,
                         ratio=1., density3=None, large=True,
                         max_ratio=4, min_ratio=0.5, **kwargs):
    """ Triple plot of amplitude, phase, irradiance

    Params
    ------
    phasor : array of shape (N, 2)
        representing amplitude, phase of N datapoints
    x,y : arrays of length N
        the corresponding 2d positions of x
    plot_func : func that takes args (x, y, color, pyplot_args)
        e.g. plt.scatter
    """
    a, phi = phasor.T
    if 'cmap' not in kwargs:
        global cmap
        kwargs['cmap'] = cmap

    n_subplots = 3
    horizontal = ratio < 1.1  # distribute subplots horizontally
    # h = 15, w = 4
    if horizontal:
        w = n_subplots * 5
        h = 4 * min(1. / ratio, max_ratio)
        if large:
            # w = n_subplots * max(ratio, min_ratio) * 7
            # h = 6
            # w = n_subplots * 7
            # h = 6 * min(1. / ratio, max_ratio)
            w *= 7. / 5.
            h *= 6. / 4.
        # else:
        #     w = n_subplots * max(ratio, min_ratio) * 5
        #     h = 4
    else:
        # h = n_subplots / min(4, ratio / 2) * 2 + 1
        h = n_subplots * 4 + 1
        # w = 8 * min(ratio, max_ratio)
        w = 5 * min(ratio, max_ratio)

    fig = plt.figure(figsize=(round(w), h))
    title_y_offset = 1.06 if horizontal else 1.04
    plt.suptitle(title, y=title_y_offset, fontsize=16, fontweight='bold')

    if horizontal:
        ax = plt.subplot(131)
    else:
        ax = plt.subplot(311)

    plot_func(x, y, a, **kwargs)
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

    plot_func(x, y, standardize(log_irradiance), **kwargs)
    if density3 is not None:
        del kwargs['density']

    markup(ax, unit='m')
    plt.title('Log Irradiance', fontsize=16)
    if horizontal:
        ax = plt.subplot(132)
    else:
        ax = plt.subplot(312)

    # cyclic cmap: hsv, twilight
    kwargs['cmap'] = cyclic_cmap
    print(plot_func.__name__)
    if plot_func.__name__ == '_hist2d_wrapper':
        plot_func(x, y, phi, range=(phi.min() / np.pi, phi.max() / np.pi),
                  **kwargs)
    else:
        plot_func(x, y, phi, **kwargs)

    markup(ax, unit='m', colorbar_unit='$\pi$')

    plt.title('Phase', fontsize=16)
    try:
        plt.tight_layout()
        # add custom subtitle after tightening layout
        if horizontal:
            x, y = 0.5, 1.075
        else:
            x, y = 0.5, 2.5
        plt.text(x, y, subtitle, {'fontsize': 12},
                 horizontalalignment='center', verticalalignment='center',
                 transform=ax.transAxes)

    except ValueError as e:
        print(e)

    if filename is not None:
        save_fig(filename, ext='png')
    return fig


def sci_labels(ax, decimals=1, x=True, y=True, z=False, unit='',
               y_unit: str = None, rotation=30):
    formatter = tck.EngFormatter(places=decimals, sep=u"\N{THIN SPACE}",
                                 unit=unit)
    if x:
        ax.xaxis.set_major_formatter(formatter)
        plt.xticks(rotation=rotation)
    if z:
        # 3D plot
        ax.zaxis.set_major_formatter(formatter)
    if y:
        if y_unit is not None:
            formatter = tck.EngFormatter(places=decimals, sep=u"\N{THIN SPACE}",
                                         unit=y_unit)
        ax.yaxis.set_major_formatter(formatter)


def markup(ax, unit='', colorbar=True, colorbar_unit='', **kwargs):
    sci_labels(ax, unit=unit, **kwargs)
    plt.xlabel("Space dimension 1")
    plt.ylabel("Space dimension 2")
    # plt.colorbar(fraction=0.052, pad=0.05,
    #              ticks=LogLocator(subs='all'), format=LogFormatterSciNotation())
    if colorbar:
        format = None
        if colorbar_unit:
            format = tck.FormatStrFormatter(f'%.2f {colorbar_unit}')
        plt.colorbar(fraction=0.052, pad=0.05, format=format)


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


def save_fig(filename, ext='png', dpi='figure',
             transparent=True, bbox_inches='tight', interpolation='none',
             **kwargs):
    assert os.path.isdir(IMG_DIR), \
        '_img_dir.py/IMG_DIR is must be setup correctly'
    # plt.axis('off') # this only affects the current subplot
    plt.savefig(f'{IMG_DIR}/{filename}.{ext}', dpi=dpi,
                transparent=transparent, interpolation=interpolation,
                bbox_inches=bbox_inches, **kwargs)
    plt.close()


def distribution1d(X, Y, title='', figshape=(1, 1), scatter=False, mean=False,
                   median=True, range=True, range_alpha=0.3, convolve=False,
                   xlog=False, ylog=True, labels=[]):
    """ X,Y: lists of x,y data """
    n_subplots = len(Y)
    cols, rows = figshape
    assert cols * rows >= n_subplots, \
        f'figsize: {cols} x {rows} < {n_subplots}'
    fig = plt.figure(figsize=(4 * cols, 3 * rows))
    for i_col, i_row in np.ndindex(figshape):
        i = i_col + i_row * cols
        if i >= n_subplots:
            continue

        plt.subplot(rows, cols, 1 + i)
        x = X[i][:, 0]
        y = Y[i]
        if scatter:
            if y.size > 100:
                plt.scatter(x, y, s=0.1, alpha=0.1)
            else:
                plt.scatter(x, y, s=5, alpha=1)

        if mean or range or median:
            n_samples = 128
            assert y.size % n_samples == 0, 'pad input to fit reduction'
            sample_size = round(y.size / n_samples)
            half_sample_size = round(sample_size / 2)  # round twice
            y_samples = y.reshape((n_samples, sample_size))
            x_reduced = x[half_sample_size::sample_size]
            assert x_reduced.shape == y_samples.mean(axis=1).shape
            assert n_samples == y_samples.mean(axis=1).size
            if mean:
                plt.plot(x_reduced, y_samples.mean(axis=1), label='mean')
            if median:
                plt.plot(x_reduced, np.median(y_samples, axis=1),
                         label='median')
            if range:
                plt.fill_between(x_reduced, y_samples.min(axis=1),
                                 y_samples.max(axis=1), alpha=range_alpha)

        if convolve and y.size > 100:
            n = round(y.size / 100)
            c = scipy.signal.hann(n)
            z = scipy.signal.convolve(y, c / c.sum(), mode='same')
            plt.plot(X[i][:, 0], z, alpha=0.8, label='convolution')

        if np.array([mean, median,  convolve]).astype(int).sum() > 1:
            plt.legend()

        if xlog:
            plt.xscale('log')
        else:
            x_min = 0 if x.min() < 1e-4 else x.min()
            plt.xlim(x_min, x.max())
        if ylog:
            plt.yscale('log')
        plt.ylim(y.min(), y.max())
        if labels:
            plt.title(labels[i], fontsize=12)
        ax = plt.gca()
        sci_labels(ax, unit='m', y=False)
        plt.xlabel('Space')
        plt.margins(0)
        plt.grid(b=None, which='major', axis='x', linewidth=0.7)
        plt.grid(b=None, which='major', linewidth=0.3, axis='y')
        plt.grid(b=None, which='minor', linewidth=0.3, axis='both')

    if title:
        plt.suptitle(title, y=1.02, fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def grid_search_result(result, x_key='rho', y_key='mean', z_keys=[],
                       err_key=None, ylog=False, standard=True,
                       interpolate=None, translate={}, baseline=None,
                       ylim=(0, None), fig=None, x_func=lambda x: x,
                       err_alpha=0.15, y_unit=None, bar=False, loc=None, v=0):
    """ Show a scatter-plot of Y as function of X,
    with conditional var Z as a third variable

    :standard: bool     interpolation with x in [0,1]. This also gives the
                        legend more space

    result = DataFrame with cols [x_key, y_key, z_key]
    """
    fit_regression_line = interpolate == 'regression'
    distinct_param_values = result.loc[:, z_keys].drop_duplicates()
    if ylog and err_key is not None:
        print("Warning, plotting (symmetric) error bars on log axis may be incorrect")

    if fig is None:
        # fig = plt.figure(figsize=(6, 4))
        fig = plt.figure(figsize=(4, 5))
    ax = plt.gca()

    for i, (_, param_values) in enumerate(distinct_param_values.iterrows()):
        if v:
            print(param_values.to_dict())
        inner = result.merge(pd.DataFrame([param_values]),
                             how='inner', right_index=True,
                             on=list(param_values.keys()))

        x = result.loc[inner.index, x_key]
        y = result.loc[inner.index, y_key]
        indices = y.values.nonzero()[0]

        x_values, y_values = x.values[indices], y.values[indices]
        x_values = x_func(x_values)
        if baseline is not None:
            y_values = np.array(baseline)[indices] / y_values

        if err_key is not None:
            err = result.loc[inner.index, err_key].values[indices]
        else:
            err = None
        if ylog:
            print('TODO check log error bars')
        # err = np.array([np.log10(err), 10 ** err]) if ylog else err

        # fit regression line

        # # log-normalize
        # if x_key == 'lapda' and z_key == 'n':
        #     # ignore invalid param values
        #     upper_bound = Params.max_lapda(n=z)
        #     indices = y.values[x.values < upper_bound].nonzero()[0]
        # if x_key == 'rho' and z_key == 'n':
        #     upper_bound = Params.max_rho()
        #     indices = y.values[x.values < upper_bound].nonzero()[0]
        # else:
        #     indices = y.values.nonzero()[0]
        #
        # if indices.size == 0:
        #     # nothing to plot
        #     break
        #
        # if ylog:
        #     y_values = np.log10(np.clip(y_values, 1e-15, None))

        if fit_regression_line:
            slope, intercept, _, p, error = scipy.stats.linregress(x_values,
                                                                   y_values)

            significant = p < 0.005

        # label = f'{translate[z_key]}: {z}'
        label = parse_label(param_values, z_keys, translate)
        # label = ', '.join(
        #     (f'{translate[k]}: {param_values[k]}' for k in z_keys))
        if fit_regression_line and significant:
            label += f' (slope: {slope:0.2f})'
        # elif fit_regression_line:
        #     print(f'Not significant! {z_key}: {z}, p:{p:0.3f}, {error:0.2f}')

        # plot scatter
        if bar:
            n_categories = distinct_param_values.shape[0]
            inter_width = 1. / 3.
            intra_width = 1 - inter_width  # group width
            width = intra_width / n_categories
            left_margin = 0.18 * n_categories / 2
            x_offset = left_margin - intra_width / 2 + i * width + width / 2
            # no need to use hatches because the order is preserved in the legend
            plt.bar(np.arange(x_values.size) + x_offset, y_values, width=width,
                    label=label, yerr=err, zorder=3)
            if i == n_categories - 1:
                # assume x_values are equal for all items
                ax.set_xticks(left_margin + np.arange(x_values.size))
                ax.set_xticklabels(x_values)

        else:
            assert i < len(COLORS)
            if err_key is None:
                plt.scatter(x_values, y_values, alpha=0.85, s=12,
                            color=COLORS[i])
            else:
                assert i < len(LINESTYLES), [i, len(LINESTYLES)]
                scale = 1
                if err_alpha > 0:
                    # don't use hatches in combination with alpha
                    plt.fill_between(x_values, y_values - err * scale,
                                     y_values + err * scale, alpha=err_alpha)

                plt.errorbar(x_values, y_values, yerr=err, fmt='x',
                             alpha=0.6, color=COLORS[i])

            plt.plot(x_values, y_values,  label=label, alpha=0.8,
                     color=COLORS[i], linestyle=LINESTYLES[i])

        # plot linear model
        if fit_regression_line and significant:
            # x_pred = np.linspace(0, 1) if standard else x.values
            if standard:
                x_pred = np.linspace(0, 1) if slope < 4 \
                    else np.linspace(0, 0.75)
            else:
                x_pred = x.values[indices]
            y_pred = slope * x_pred + intercept
            if ylog:
                y_pred = 10 ** y_pred

            plt.plot(x_pred, y_pred, '-', alpha=0.5,
                     linewidth=1, color=COLORS[i])

        elif interpolate is True:
            assert interpolate != 'regression', 'incorrect arg'
            plt.plot(x, y, '-', alpha=0.5,
                     linewidth=1, color=COLORS[i])

    # Add markup
    plt.xlabel(translate[x_key])
    plt.ylabel(translate[y_key])
    sci_labels(ax, x=False, y_unit=y_unit)
    plt.legend(loc=loc)
    # plt.xlim(xlim)
    plt.ylim(ylim)
    if ylog:
        plt.yscale('log')
        # plt.yscale('symlog')

    # add grid
    if not bar:
        plt.grid(b=None, which='major', axis='x', linewidth=0.7)
    plt.grid(b=None, which='major', linewidth=0.3, axis='y')
    plt.grid(b=None, which='minor', linewidth=0.3, axis='y' if bar else 'both')
    if not ylog and not bar:
        ax.xaxis.set_minor_locator(tck.AutoMinorLocator())  # TODO tst
        ax.yaxis.set_minor_locator(tck.AutoMinorLocator())  # TODO tst

    # rm spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    return fig


def parse_label(param_values, z_keys, translate: dict):
    segments = []
    for k in z_keys:
        value = param_values[k]
        # translate value
        if value in translate.keys():
            value = translate[value]
        else:
            if isinstance(value, float):
                if abs(value) > 1e-3:
                    value = round(value, 3)
                else:
                    value = f'{value:0.1e}'

        if k == 'algorithm':
            value = f'#{value}'

        # translate label
        segments.append(f'{translate[k]}: {value}' if translate[k] else value)
    return ', '.join(segments)


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
    ratio = params['y'][0]['aspect_ratio']
    Nx, Ny = util.solve_xy_is_a(N, ratio)
    Nxy = Nx * Ny
    for k in 'yv':
        data[k] = data[k][: Nxy]

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
            matrix(reshape(z[:, 0], False), f'z_{i} Amplitude')
            plt.show()
