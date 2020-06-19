import numpy as np
import subprocess
import os
# local
import plot
import surf
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
        for i in range(len(data[k])):
            data[k][i] = data[k][i][:Nxy]

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
    pointsize = 0.1  # for large scatter plots
    # indices = np.arange(N).reshape((N_sqrt, N_sqrt))[:n, :n].flatten()
    m = len(data['x'])
    args = (m,) if m <= 4 else (0, m, np.ceil(m / 4).astype(int))
    for i in range(*args):
        # offset = params['x'][i]['z_offset']
        # title = f"$x_{{{i}}}$ (distance: {round(offset, 2)} m)"
        # title = plot.format_title('x', i, offset)
        # title = format_title('x', i, params['x'][i]['z_offset'])
        subtitle = f"(distance: {round(params['x'][i]['z_offset'], 2)} m)"
        plot.scatter_multiple(data['x'][i][:n], data['u'][i][:n],
                              f'Object ({i})', subtitle, filename=f'x-scatter-sub-{i}')

        # offset = params['y'][i]['z_offset']
        # title = f"$y_{{{i}}}$ (distance: {round(offset, 2)} m)"
        subtitle = f"(distance: {params['y'][i]['z_offset']:0.2f} m)"
        # title = format_title('y', i, params['y'][i]['z_offset'])
        # indices = np.random.randint(0, Nxy, n)
        # plot.scatter_multiple(data['y'][i][indices], data['v'][i][indices],
        #                       title, filename=f'y-scatter-sub-{i}', s=pointsize)

        plot.hist_2d_multiple(data['y'][i], data['v'][i],
                              f'Projector ({i})', subtitle,
                              filename=f'y-hist2d-{i}',
                              ybins=bins, ratio=ratio)

    print('plot various z')
    N = data['z'][0].shape[0]
    ratio = 1920. / 1080. if params['z'][0]['hd'] else 1.
    Nx, Ny = util.solve_xy_is_a(N, ratio)
    Nxy = Nx * Ny
    for k in 'yv':
        for i in range(len(data[k])):
            data[k][i] = data[k][i][:Nxy]
    if 'z' in params.keys() and params['z'][0]['hd']:
        ratio = 1920. / 1080.
    else:
        ratio = 1.
    bins = int(min(N_sqrt, 1080))
    n_z_per_y = len(data['z']) // len(data['y'])
    # for i in range(min(n_z_plots, len(data['z']))):
    #     major = i // len(data['x'])
    #     minor = i % len(data['x'])
    m = len(data['y'])
    args1 = (m,) if m <= 5 else (0, m, np.ceil(m / 5).astype(int))
    m = n_z_per_y
    args2 = (m,) if m <= 5 else (0, m, np.ceil(m / 5).astype(int))
    for major in range(*args1):
        for minor in range(*args2):
            i = major * n_z_per_y + minor
            # offset = params['z'][i]['z_offset']
            title = f"Projection {minor} (Object {major})"
            # title = f"$z_{{{major}, {minor}}}$ " + \
            #     f"(distance: {round(offset, 2)} m)"
            # title = format_title('z', f"{major, minor}",
            #                      params['z'][i]['z_offset'])
            subtitle = f"(distance: {round(params['z'][i]['z_offset'], 2)} m)"

            N = data['z'][i].shape[0]
            N_sqrt = np.sqrt(N).astype(int)
            # indices = np.random.randint(0, N, n)
            # fn = f'z-scatter-sub-{major}-{minor}'
            # plot.scatter_multiple(data['z'][i][indices], data['w'][i][indices],
            #                       title, filename=fn, s=pointsize)

            fn = f'z-hist2d-{major}-{minor}'
            plot.hist_2d_multiple(data['z'][i], data['w'][i],
                                  title, subtitle,
                                  filename=fn, ybins=bins,
                                  ratio=ratio)

            fn = f'z-surf-{major}-{minor}'
            # TODO add suptitle, title
            surf.surf_multiple(data['z'][i], data['w'][i], N_sqrt, N_sqrt,
                               f'{title}\n{subtitle}\n', filename=fn)

    print('plot sequence z')
    bins = int(min(N_sqrt, 1000))
    minors = list(util.pendulum(n_z_per_y))
    for major in range(len(data['y'])):
        for i, minor in enumerate(minors):
            j = major * n_z_per_y + minor  # idx in data['z']
            frame_idx = major * len(minors) + i
            amp = data['z'][j][:, 0]
            plot.hist_2d_hd(amp ** 2, data['w'][j],
                            filename=f'{sequence_dir}z/{frame_idx:06}', ybins=bins,
                            ratio=ratio, verbose=0)

    # bins = min(1080, Ny)
    # animate.multiple_hd(
    #     ratio, data['y'], data['v'], prefix='y-ani', ybins=bins)

    # animate.multiple(data['z'], data['w'], prefix='z-ani', bins=bins,
    #                  offsets=[p['z_offset'] for p in params['z']])
