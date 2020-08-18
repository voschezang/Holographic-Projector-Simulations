import numpy as np
import subprocess
import os
import matplotlib.pyplot as plt
# local
import plot
import surf
import animate
import util


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
    if size > 1e8:
        print(f'Warning, file too large: {size*1e-6:0.4f} MB')

    params, data = util.parse_file(dir, fn, 'out')
    # print('uu', data['u'])

    N = data['y'][0].shape[0]
    ratio = params['y'][0]['aspect_ratio']
    Nx, Ny = util.solve_xy_is_a(N, ratio)
    Nxy = Nx * Ny
    print({'N': N, 'Nx': Nx, 'Ny': Ny, 'eq': Nx * Ny == N})
    for k in 'yv':
        for i in range(len(data[k])):
            data[k][i] = data[k][i][:Nxy]

    i = 0
    # bins = min(1080, Ny)
    bins = Ny
    print(f'bw plots ({bins}/1080 ybins)')
    # plot.hist_2d_hd(data['y'][i], data['v'][i],
    #                 filename=f'y-hist2d', ybins=bins, ratio=ratio)

    # plot.hist_2d_hd(data['y'][i], data['v'][i],
    #                 filename=f'y-hist2d-lo', ybins=bins / 4,
    #                 ratio=ratio)

    if 1:
        i = 0
        plot.hist_2d_hd(data['y'][i], data['v'][i],
                        filename=f'y_{i}', ybins=bins,
                        ratio=ratio, bin_options=params['y'][i], verbose=0)
    else:
        for i in range(len(data['y'])):
            print(
                f"plot y_{i}, mu: {data['y'][i][:,0].mean()}, {data['y'][i][:,1].mean()}")
            plot.hist_2d_hd(data['y'][i], data['v'][i],
                            filename=f'y_{i}', ybins=bins,
                            ratio=ratio, bin_options=params['y'][i], verbose=0)

    print('plot sequence y')
    for j, i in enumerate(util.pendulum(len(data['y']))):
        amp = data['y'][i][:, 0]
        amp /= amp.max()
        print('amp max', max(amp))
        plot.hist_2d_hd(amp ** 2, data['v'][i],
                        filename=f'{sequence_dir}y/{j:06}', ybins=bins,
                        ratio=ratio, bin_options=params['y'][i], verbose=0)

    print('plot various x, y')
    n = int(5e3)
    n = int(1e4)
    pointsize = 0.1  # for large scatter plots
    m = len(data['x'])
    assert(len(data['x']) == len(data['y']))
    args = (m,) if m <= 4 else (0, m, np.ceil(m / 4).astype(int))
    for i in range(*args):
        subtitle = f"(distance: {round(params['x'][i]['z_offset'], 2)} m)"
        plot.scatter_multiple(data['x'][i][:n], data['u'][i][:n],
                              f'Object ({i})', subtitle, filename=f'x-scatter-sub-{i}')

        subtitle = f"(distance: {params['y'][i]['z_offset']:0.2f} m)"
        if 0:
            indices = np.random.randint(0, Nxy, n) \
                if n < Nxy else np.arange(Nxy)
            plot.scatter_multiple(data['y'][i][indices], data['v'][i][indices],
                                  f'Projector ({i})',
                                  filename=f'y-scatter-sub-{i}', s=pointsize)

        plot.hist_2d_multiple(data['y'][i], data['v'][i],
                              f'Projector ({i})', subtitle,
                              filename=f'y-hist2d-{i}',
                              ybins=bins, ratio=ratio, bin_options=params['y'][i])

    if 'z' in data.keys() and len(data['z']):
        print('plot various z')
        N = data['z'][0].shape[0]
        ratio = params['z'][0]['aspect_ratio']
        Nx, Ny = util.solve_xy_is_a(N, ratio)
        Nxy = Nx * Ny
        for k in 'zw':
            for i in range(len(data[k])):
                data[k][i] = data[k][i][:Nxy]

        n_z_per_y = len(data['z']) // len(data['y'])
        m = len(data['y'])
        args1 = (m,) if m <= 5 else (0, m, np.ceil(m / 5).astype(int))
        m = n_z_per_y
        args2 = (m,) if m <= 5 else (0, m, np.ceil(m / 5).astype(int))
        for major in range(*args1):
            for minor in range(*args2):
                i = major * n_z_per_y + minor
                print('plot ', i)
                title = f"Projection {minor} (Object {major})"
                subtitle = f"(distance: {round(params['z'][i]['z_offset'], 2)} m)"
                if 0:
                    indices = np.random.randint(0, Nxy, n) \
                        if n < Nxy else np.arange(Nxy)
                    fn = f'z-scatter-sub-{major}-{minor}'
                    plot.scatter_multiple(data['z'][i][indices], data['w'][i][indices],
                                          title, subtitle=subtitle,
                                          filename=fn, s=pointsize)

                fn = f'z-hist2d-{major}-{minor}'
                plot.hist_2d_multiple(data['z'][i], data['w'][i],
                                      title, subtitle,
                                      filename=fn, ybins=Ny, ratio=ratio,
                                      bin_options=params['z'][i])
                if 0:
                    plot.hist_2d_multiple(data['z'][i], data['w'][i],
                                          title, subtitle,
                                          filename=fn + '_nobins', ybins=Ny, ratio=ratio)

                fn = f'z-surf-{major}-{minor}'
                # TODO add suptitle, title to surf
                surf.surf_multiple(data['z'][i], data['w'][i], Nx, Ny,
                                   f'{title}\n{subtitle}\n', filename=fn)

        print('plot sequence z')
        minors = list(util.pendulum(n_z_per_y))
        for major in range(len(data['y'])):
            for i, minor in enumerate(minors):
                j = major * n_z_per_y + minor  # idx in data['z']
                frame_idx = major * len(minors) + i
                amp = data['z'][j][:, 0]
                plot.hist_2d_hd(amp ** 2, data['w'][j],
                                filename=f'{sequence_dir}z/{frame_idx:06}',
                                ybins=Ny, ratio=ratio,
                                bin_options=params['z'][j], verbose=0)

        # bins = min(1080, Ny)
        # animate.multiple_hd(
        #     ratio, data['y'], data['v'], prefix='y-ani', ybins=bins)

        # animate.multiple(data['z'], data['w'], prefix='z-ani', bins=bins,
        #                  offsets=[p['z_offset'] for p in params['z']])
