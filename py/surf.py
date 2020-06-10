import numpy as np
import subprocess
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

# local
import plot
import animate
import util
from util import DIMS


def run():
    return subprocess.check_output(['make', 'build-run'])


def surf(x, y, z, Nx: int, Ny: int, **kwargs):
    if 'cmap' not in kwargs:
        global cmap
        kwargs['cmap'] = plot.cmap

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    X = x.reshape((Nx, Ny))
    Y = y.reshape((Nx, Ny))
    Z = z.reshape((Nx, Ny))

    surf = ax.plot_surface(X, Y, Z, linewidth=0, antialiased=False, **kwargs)
    return surf


def surf_multiple(phasor, position, prefix=''):
    labels = ['Amplitude^2', 'Phase', 'Log Irradiance']
    for i, label in enumerate(labels):
        if i == 1:
            # skip phase plots
            continue

        if i < 2:
            # square amp
            z = phasor[:, i]
            if i == 1:
                z = z ** 2
        else:
            a, phi = phasor.T
            log_irradiance = np.log(util.irradiance(
                util.to_polar(a, phi), normalize=False))
            z = util.standardize(log_irradiance)
            assert abs(z.min()) < 1e-3
            assert abs(1 - z.max()) < 1e-3
            print('z range', z.min(), z.max())

        # ignore third dimension
        surf(position[:, 0], position[:, 1],
             z, Nx, Ny)
        plt.title(f'{prefix}{label}')
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    n_z_plots = 10
    sequence_dir = util.get_arg('--sequence_dir', '', parse_func=str)
    if util.get_flag("-r") or util.get_flag("--rerun"):
        out = run()

    log = util.get_flag("-log")
    if log:
        print('log abs y')

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
    # max_n_plots = 2
    # m = len(data['y'])
    # args = (m,) if m <= max_n_plots else (
    #     0, m, np.ceil(m / max_n_plots).astype(int))
    # for i in range(*args):
    #     surf_multiple(data['y'][i], data['v'][i], f'$y_{i} $')

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
            surf_multiple(data['z'][i], data['w'][i], prefix)