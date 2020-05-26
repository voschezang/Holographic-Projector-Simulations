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
    n_z_plots = 2
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
        print(f'WARNING, file too large: {size*1e-6:0.4f} MB')

    params, data = util.parse_file(dir, fn, 'out')

    # if util.get_flag("-scatter"):
    #     plot.matrix_multiple(data['x'], 'x', filename='x')
    #     # plot.scatter_multiple(data['x'], data['u'], 'x', filename='x')
    #     plot.scatter_multiple(data['y'], data['v'], 'y', filename='y', s=1)
    #     k = 'z'
    #     if k in data.keys():
    #         plot.scatter_multiple(data['z'], data['w'], k, filename=k)
    # else:
    #     plot.matrix_multiple(data['x'], 'x', filename='x')
    #     plot.matrix_multiple(data['y'], 'y', filename='y')
    #
    #     k = 'z'
    #     if k in data.keys():
    #         plot.matrix_multiple(data[k], k, filename=k)

    # n = 100
    # plot.scatter_multiple(data['x'][:n], data['u'][:n],
    #                       'x', filename='x-scatter-sub', s=1)

    # plot subset
    N = data['y'].shape[0]
    ratio = 1920. / 1080. if params[1]['hd'] else 1.
    Nx, Ny = util.solve_xy_is_a(N, ratio)
    Nxy = Nx * Ny
    for k in 'yv':
        data[k] = data[k][:Nxy]

    print({'N': N, 'Nx': Nx, 'Ny': Ny, 'eq': Nx * Ny == N})
    # print(N, Nx, Ny, f'equal: {Nx * Ny == N}')
    # print(f'Nxy {Nxy}')
    N_sqrt = int(np.sqrt(N))
    print(f'N sqrt (y): {N_sqrt}')

    bins = min(1080, Ny)
    print(f'bw plots ({bins}/1080 ybins)')
    print('y,v pos ranges:',
          f"{data['v'][:, 0].max() - data['v'][:, 0].min():.4E}",
          f"{data['v'][:, 1].max() - data['v'][:, 1].min():.4E}")
    plot.hist_2d_hd(data['y'], data['v'],
                    cmap='gray', filename='y-hist2d', ybins=bins, ratio=ratio)

    plot.hist_2d_hd(data['y'], data['v'],
                    cmap='gray', filename='y-hist2d-lo', ybins=bins / 4, ratio=ratio)

    print('sample scatters')
    n = int(1e4)
    # indices = np.arange(N).reshape((N_sqrt, N_sqrt))[:n, :n].flatten()
    plot.scatter_multiple(data['x'][:n], data['u'][:n],
                          f"x (offset: {params[0]['z_offset']} m)",
                          filename='x-scatter-sub')

    indices = np.random.randint(0, Nxy, n)
    plot.scatter_multiple(data['y'][indices], data['v'][indices],
                          f"y (offset: {params[1]['z_offset']} m)",
                          filename='y-scatter-sub', s=1)

    for i in range(min(n_z_plots, len(data['z']))):
        N = data['z'][i].shape[0]
        N_sqrt = np.sqrt(N).astype(int)
        print(f'N sqrt (z_{i}): {N_sqrt}')
        indices = np.random.randint(0, N, n)
        title = f"$z_{i}$ (offset: {params[2]['z_offset']} m)"
        plot.scatter_multiple(data['z'][i][indices], data['w'][i][indices],
                              title, filename=f'z-scatter-sub-{i}', s=1)

    # gridsize = round(max(25, N / 5e2))
    # bins = int(round(N_sqrt / 2))
    # print(f'hexbin: N^2: {n}, grid: {gridsize}')
    # plot.hexbin_multiple(data['y'], data['v'], 'y',
    #                      filename=f'y-hexbin', bins=bins)
    # # plot.hexbin_multiple(data['z'][indices], data['w'][indices], 'z',
    # #                      filename=f'z-hexbin', gridsize=gridsize)

    plot.hist_2d_multiple(data['y'], data['v'],
                          f"y (offset: {params[1]['z_offset']} m)",
                          filename='y-hist2d', ybins=bins / 2, ratio=ratio)

    bins = int(min(N_sqrt, 1000))
    for i in range(min(n_z_plots, len(data['z']))):
        title = f"$z_{i}$ (offset: {params[2]['z_offset']} m)"
        plot.hist_2d_multiple(data['z'][i], data['w'][i], title,
                              filename=f'z-hist2d-{i}', ybins=bins,
                              ratio=1.)
    # w = data['w'][0]
    # print(w.shape)
    # tmp = w[:, 0].copy()
    # w[:, 0] = w[:, 1]
    # w[:, 1] = tmp
    # print(w.shape)
    # plot.hist_2d_multiple(data['z'][0], w,
    #                       f"z (offset: {params[2]['z_offset']} m)",
    #                       filename='z-hist2d-T', ybins=bins, ratio=1.)

    # animate.multiple(data['z'], data['w'], prefix='z-ani', bins=bins,
    #                  offsets=[p['z_offset'] for p in params[2:]])
