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
    Nx = int(np.sqrt(N * ratio))
    Ny = N // Nx
    Nxy = Nx * Ny
    for k in 'yv':
        data[k] = data[k][:Nxy]

    print(N, Nx, Ny, f'equal: {Nx * Ny == N}')
    print(f'Nxy {Nxy}')
    N_sqrt = np.sqrt(N).astype(int)
    print(f'N sqrt: {N_sqrt}')

    print('bw plots')
    bins = min(1920, Nx)
    plot.hist_2d_hd(data['y'][:Nxy], data['v'][:Nxy],
                    cmap='gray', filename='y-hist2d', xbins=bins, ratio=ratio)

    plot.hist_2d_hd(data['y'][:Nxy], data['v'][:Nxy],
                    cmap='gray', filename='y-hist2d-lo', xbins=Nx // 4, ratio=ratio)

    if 1:
        print('sample scatters')
        # n = int(5e3)
        n = int(5e3)
        n = int(1e4)
        # indices = np.arange(N).reshape((N_sqrt, N_sqrt))[:n, :n].flatten()
        indices = np.random.randint(0, Nxy, n)
        # indices = np.arange(N)
        plot.scatter_multiple(data['x'][:n], data['u'][:n],
                              f"x (offset: {params[0]['z_offset']} m)",
                              filename='x-scatter-sub')
        plot.scatter_multiple(data['y'][indices], data['v'][indices],
                              f"y (offset: {params[1]['z_offset']} m)",
                              filename='y-scatter-sub', s=1)
        i = 0
        plot.scatter_multiple(data['z'][i][indices], data['w'][i][indices],
                              f"z (offset: {params[2]['z_offset']} m)",
                              filename='z-scatter-sub', s=1)

        # gridsize = round(max(25, N / 5e2))
        # bins = int(round(N_sqrt / 2))
        # print(f'hexbin: N^2: {n}, grid: {gridsize}')
        # plot.hexbin_multiple(data['y'], data['v'], 'y',
        #                      filename=f'y-hexbin', bins=bins)
        # # plot.hexbin_multiple(data['z'][indices], data['w'][indices], 'z',
        # #                      filename=f'z-hexbin', gridsize=gridsize)

        # TODO find optimal number of bins for datasize
        N = data['z'][0].shape[0]
        N_sqrt = np.sqrt(N).astype(int)
        print(f'N sqrt: {N_sqrt}')
        bins = int(round(N_sqrt / 2))
        plot.hist_2d_multiple(data['y'], data['v'],
                              f"y (offset: {params[1]['z_offset']} m)",
                              filename='y-hist2d', bins=bins)
        plot.hist_2d_multiple(data['z'][0], data['w'][0],
                              f"z (offset: {params[2]['z_offset']} m)",
                              filename='z-hist2d', bins=bins)

        print([p['z_offset'] for p in params[2:]])
        animate.multiple(data['z'], data['w'], prefix='z-ani', bins=bins,
                         offsets=[p['z_offset'] for p in params[2:]])
