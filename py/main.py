import numpy as np
import subprocess
import zipfile
import os
import matplotlib.pyplot as plt
# local
import plot
import animate
import util
from util import DIMS


def run():
    return subprocess.check_output(['make', 'build-run'])


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
    # print(N, Nx, Ny, f'equal: {Nx * Ny == N}')
    # print(f'Nxy {Nxy}')
    N_sqrt = int(np.sqrt(N))
    print(f'N sqrt (y): {N_sqrt}')

    i = 0
    bins = min(1080, Ny)
    print(f'bw plots ({bins}/1080 ybins)')
    print('y,v pos ranges:',
          f"{data['v'][i][:, 0].max() - data['v'][i][:, 0].min():.4E}",
          f"{data['v'][i][:, 1].max() - data['v'][i][:, 1].min():.4E}")

    # plot.hist_2d_hd(data['y'][i], data['v'][i],
    #                 filename=f'y-hist2d', ybins=bins, ratio=ratio)

    # plot.hist_2d_hd(data['y'][i], data['v'][i],
    #                 filename=f'y-hist2d-lo', ybins=bins / 4,
    #                 ratio=ratio)

    print('plot sequence y')
    for i, j in enumerate(util.pendulum(len(data['y']))):
        amp = data['y'][j][:, 0]
        plot.hist_2d_hd(amp ** 2, data['v'][j],
                        filename=f'{sequence_dir}y/{i:06}', ybins=bins,
                        ratio=ratio, verbose=0)

    print('plot various y')
    n = int(5e3)
    # indices = np.arange(N).reshape((N_sqrt, N_sqrt))[:n, :n].flatten()
    m = len(data['x'])
    args = (m,) if m <= 4 else (0, m, np.ceil(m / 4).astype(int))
    for i in range(*args):
        title = f"$x_{{{i}}}$ (offset: {params['x'][i]['z_offset']:03f} m)"
        plot.scatter_multiple(data['x'][i][:n], data['u'][i][:n],
                              title, filename=f'x-scatter-sub-{i}')

        indices = np.random.randint(0, Nxy, n)
        plot.scatter_multiple(data['y'][i][indices], data['v'][i][indices],
                              title, filename=f'y-scatter-sub-{i}', s=1)

        plot.hist_2d_multiple(data['y'][i], data['v'][i],
                              title, filename=f'y-hist2d-{i}',
                              ybins=bins / 2, ratio=ratio)

    print('plot various z')
    bins = int(min(N_sqrt, 1000))
    n_z_per_y = len(data['z']) // len(data['y'])
    # for i in range(min(n_z_plots, len(data['z']))):
    #     major = i // len(data['x'])
    #     minor = i % len(data['x'])
    m = len(data['y'])
    args1 = (m,) if m <= 4 else (0, m, np.ceil(m / 4).astype(int))
    m = n_z_per_y
    args2 = (m,) if m <= 4 else (0, m, np.ceil(m / 4).astype(int))
    for major in range(*args1):
        for minor in range(*args2):
            i = major * n_z_per_y + minor
            title = f"$z_{{{major}, {minor}}}$ " + \
                f"(offset: {params['z'][i]['z_offset']:03f} m)"

            N = data['z'][i].shape[0]
            N_sqrt = np.sqrt(N).astype(int)
            indices = np.random.randint(0, N, n)
            fn = f'z-scatter-sub-{major}-{minor}'
            plot.scatter_multiple(data['z'][i][indices], data['w'][i][indices],
                                  title, filename=fn, s=1)

            fn = f'z-hist2d-{major}-{minor}'
            plot.hist_2d_multiple(data['z'][i], data['w'][i], title,
                                  filename=fn, ybins=bins,
                                  ratio=1.)

    print('plot sequence z')
    bins = int(min(N_sqrt, 1000))
    minors = list(util.pendulum(n_z_per_y))
    for major in range(len(data['y'])):
        for i, minor in enumerate(minors):
            j = major * n_z_per_y + minor  # idx in data['z']
            frame_idx = major * len(minors) + i
            amp = data['z'][j][:, 0]
            plot.hist_2d_hd(amp, data['w'][j],
                            filename=f'{sequence_dir}z/{frame_idx:06}', ybins=bins,
                            ratio=1., verbose=0)

    # bins = min(1080, Ny)
    # animate.multiple_hd(
    #     ratio, data['y'], data['v'], prefix='y-ani', ybins=bins)

    # animate.multiple(data['z'], data['w'], prefix='z-ani', bins=bins,
    #                  offsets=[p['z_offset'] for p in params['z']])
