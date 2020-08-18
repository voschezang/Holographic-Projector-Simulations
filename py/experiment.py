import numpy as np
import subprocess
import collections
import matplotlib.pyplot as plt
import scipy.stats
import scipy.optimize

# local
import util
import remote
import plot
import surf

EXE = 'run_experiment'
CD = 'cd ../cuda'
DIR = 'tmp_local'

translate = collections.defaultdict(lambda: 'value unknown')
translate.update({'Runtime mean': 'Runtime',
                  'FLOPS mean': 'FLOPS',
                  'kernel_size': 'Kernel size',
                  'n_objects': 'N',
                  'n_pixels': 'M',
                  'N': r'$\sqrt{N\cdot M}$',
                  'n_streams': 'S',
                  'algorithm': '',
                  'thread_size': 'T',
                  'blockDim_x': 'B.x',
                  'blockDim_y': 'B.y',
                  'gridDim_x': 'G.x',
                  'gridDim_y': 'G.y',
                  })
translate_latex = translate.copy()
translate_latex['n_objects'] = 'N'


def build():
    flags = '-l curand -l cublas -std=c++14'
    arch_flags = '-arch=compute_70 -code=sm_70'
    # macros = []
    # -D{' -D'.join(macros)}
    content = f"""
{remote.SSH} << EOF
source ~/.profile
cd cuda
nvcc -o {EXE} main.cu {flags} {arch_flags}
EOF
"""
    return subprocess.run(content, shell=True, check=True, capture_output=True)


def run(n_objects=1, n_pixels=1920 * 1080, n_sample_points=None, N=None,
        X=1, Z=0,
        obj_x_offset_min=0,
        obj_z_offset_min=1e-2, obj_z_offset_max=None,
        projection_z_offset_min=0., projection_z_offset_max=None,
        projection_width=None,
        aspect_ratio_projector=1., aspect_ratio_projection=1.,
        algorithm=1, quadrant_projection=False, randomize_pixels=False,
        n_streams=16, thread_size=(4, 4),
        blockDim_x=16, blockDim_y=16,
        gridDim_x=16, gridDim_y=16):
    assert N is not None or n_objects is not None
    assert X >= 1
    if obj_z_offset_max is None:
        obj_z_offset_max = obj_z_offset_min
    if projection_z_offset_max is None:
        projection_z_offset_max = projection_z_offset_min
    if N is not None:
        n_objects = N
        n_pixels = N
    if n_pixels is None:
        n_pixels = n_objects
    if n_sample_points is None:
        n_sample_points = n_pixels
    if projection_width is None:
        projection_width = n_sample_points * 7e-6
    # projector_width = y * 7e-6 # TODO
    assert blockDim_x * blockDim_y <= 1204, 'Max number of threads per block'

    # TODO catch errors caused by "None" values in flags at program-runtime
    for i, v in enumerate([n_objects, n_pixels, n_sample_points, X, Z,
                           aspect_ratio_projector, aspect_ratio_projection,
                           obj_x_offset_min,
                           obj_z_offset_min, obj_z_offset_max,
                           projection_width,
                           projection_z_offset_min, projection_z_offset_max,
                           algorithm, n_streams,
                           thread_size[0], thread_size[1],
                           blockDim_x, blockDim_y,
                           gridDim_x, gridDim_y]):
        assert v is not None
        if not v:
            print(f'Warning, param {i}:{v} may be incorrect')

    flags = [f'-x {n_objects} -y {n_pixels} -z {n_sample_points}',
             f'-X {X} -Z {Z}',
             f'-a {aspect_ratio_projector} -A {aspect_ratio_projection}',
             f'-u {obj_x_offset_min} -v 0',
             f'-w {obj_z_offset_min} -W {obj_z_offset_max}',
             f'-o 0. -O 0.',
             f'-n {projection_width}',
             f'-m {projection_z_offset_min} -M {projection_z_offset_max}',
             f'-p {algorithm:d}',
             f'-q' * quadrant_projection,  # bool flag
             f'-r' * randomize_pixels,  # bool flag
             f'-s {n_streams}',
             f'-t {thread_size[0]} -T {thread_size[1]}',
             f'-b {blockDim_x} -B {blockDim_y}',
             f'-g {gridDim_x} -G {gridDim_y}',
             ]
    content = f"""
{remote.SSH} << EOF
source ~/.profile
cd cuda
make cleanup-output
./{EXE} {' '.join(flags)}
make zip
EOF
"""
    print('content', content, '\n')
    return subprocess.run(content, shell=True, check=True, capture_output=True)


def performance(n_trials: int, fn: str, v=0):
    build_param_values = {}
    run_param_values = {
        # 'N': np.array([256, 128, 512]) ** 2,
        'n_objects': np.array([128, 256, 512]) ** 2,
        'n_pixels': [128 ** 2],
        'algorithm': [1, 2, 3],
        # 'blockDim_x': np.logspace(3, 5, 3, base=2, dtype=int),
        # 'blockDim_y': [2, 4],
        # 'blockDim_y': np.logspace(1, 3, 3, base=2, dtype=int),
        'n_streams': [16],
        # 'thread_size': [(2, 2), (16, 1), (1, 16), (16, 16)],
        'thread_size': [(8, 8), (16, 16)],
        'blockDim_x': [16, 32],
        'blockDim_y': [8, 16],
        'gridDim_x': [8],
        'gridDim_y': [8]}
    build_params = util.param_table(build_param_values)
    run_params = util.param_table(run_param_values)
    print('build_params', build_param_values)
    print('run_params', run_param_values)

    results, results_fn = util.get_results(build, run, build_params, run_params, fn,
                                           n_trials=n_trials, v=v)
    # results = build_params.join(run_params).join(results)
    # params, = util.parse_file(dir, f'{results_fn}.zip', 'out', read_data=False)
    # print(params)
    # print(results.head())
    # selection = results
    # selection = results.query('blockDim_y == 8')
    # selection = results.query('gridDim_y == 2')
    # selection = results.query('n_streams == 16')
    for metric in ['Runtime', 'FLOPS']:
        y_key = f'{metric} mean'
        maxi = results.loc[:, y_key].max()
        print(metric, maxi)
        for a in [1, 2, 3]:
            # selection = results.query(f'blockDim_y == {b}')
            # selection = results.query(f'algorithm == {a} and blockDim_x == 16')
            selection = results.query(f'algorithm == {a}')
            for bar in [1]:
                fig = plt.figure(figsize=(7, 4))
                plot.grid_search_result(selection, x_key='n_objects',
                                        y_key=y_key,
                                        z_keys=[
                                            # 'algorithm',
                                            'thread_size',
                                            'blockDim_x',
                                            'blockDim_y',
                                            # 'n_objects',
                                            # 'n_pixels',
                                            # 'gridDim_x'
                                        ],
                                        err_key=f'{metric} std',
                                        translate=translate,
                                        # ylog=metric == 'FLOPS',
                                        ylog=0,
                                        y_unit='s' if metric == 'Runtime' else None,
                                        bar=bar,
                                        interpolate=0,
                                        # loc='lower right',
                                        loc='upper left',
                                        fig=fig,
                                        # interpolate=0 if bar else 'regression',
                                        err_alpha=0., v=v)
                # plt.ylim((0, None))
                suffix = f'\n(Algorithm {a})'
                if metric == 'FLOPS':
                    # hlim = 1e11  # 7.4e12
                    # plt.axhline(hlim, label='Theoretical limit', ls='--', lw=1,
                    #             alpha=0.4, color='0')
                    # plt.ylim(0, hlim * 1.01)
                    # plt.legend()
                    plt.title(f'Efficiency{suffix}')
                else:
                    plt.title(f'{metric}{suffix}')
                plt.ylim(0, maxi * 1.05)
                plt.tight_layout()
                # plot.save(f'runtime-{bar}')
                plot.save_fig(f'{fn}-{metric}-{bar}-{a}', transparent=False)


def distribution_vary_object(fn: str, n: int, projection_width=1, v=0):
    build_param_values = {}
    run_param_values = {'obj_z_offset_min': 0.01,
                        'obj_z_offset_max': 1,
                        'projection_width': projection_width,
                        'aspect_ratio_projection': n,
                        'quadrant_projection': True,
                        # 'randomize_pixels': True,
                        'y': n, 'X': 4, 'Z': 1}
    build_params = util.param_table(build_param_values)
    run_params = util.param_table(run_param_values)
    print('build_params', build_param_values)
    print('run_params', run_param_values)

    results, results_fn = util.get_results(build, run, build_params, run_params, fn,
                                           n_trials=1, tmp_dir=DIR,
                                           copy_data=True, v=v)

    print('load')
    params, data = util.parse_file(DIR, f'{results_fn}.zip', 'out')
    print('post')

    for xlog, ylog in np.ndindex((1, 2)):
        title = ('Log ' * ylog) + 'Amplitude Near Projector'
        plot.distribution1d(data['w'], [z[:, 0] for z in data['z']], title,
                            xlog=xlog, ylog=ylog,
                            figshape=(2, 2),
                            labels=[f"Distance: ${p['z_offset']:.3f}$ m"
                                    for p in params['z']])
        plot.save_fig(f'{fn}-amp-{xlog}{ylog}', transparent=False)


def distribution_vary_projector(fn: str, n_samples: int, v=0):
    build_param_values = {}
    obj_z_offset = 0.2
    # abs_projection_offset = 0.1 * obj_z_offset
    abs_projection_offset = 5e-3
    m = 5
    run_param_values = {'obj_z_offset_min': obj_z_offset,
                        'projection_z_offset_min': -abs_projection_offset,
                        'projection_z_offset_max': +abs_projection_offset,
                        'aspect_ratio_projection': n_samples,
                        'quadrant_projection': True,
                        # 'randomize_pixels': True,
                        'y': n_samples, 'X': 1, 'Z': m}
    build_params = util.param_table(build_param_values)
    run_params = util.param_table(run_param_values)
    print('build_params', build_param_values)
    print('run_params', run_param_values)

    results, results_fn = util.get_results(build, run, build_params, run_params, fn,
                                           n_trials=1, tmp_dir=DIR,
                                           copy_data=True, v=v)

    print('load')
    params, data = util.parse_file(DIR, f'{results_fn}.zip', 'out')
    print('post')
    if m > 4:
        # change order of items
        assert m % 2 == 1
        half = m // 2
        for seq in [params['z'], data['z'], data['w']]:
            for i in range(half):
                seq.insert(1 + 2 * i, seq.pop(-1))

    labels = [f"Distance: ${p['z_offset'] - obj_z_offset:.4f}$ m"
              for p in params['z']]
    print('labels', labels)

    for xlog, ylog in np.ndindex((1, 1)):
        title = ('Log ' * ylog) + 'Amplitude Near Object'
        plot.distribution1d(data['w'], [z[:, 0] for z in data['z']], title,
                            xlog=xlog, ylog=ylog,
                            figshape=(2, 3), labels=labels)
        plot.save_fig(f'{fn}-amp-{xlog}{ylog}', transparent=False)


def fit_gaussian(fn: str):
    # n_sqrt = 512
    n_sqrt = 256
    build_param_values = {}
    run_param_values = {
        'n_objects': 1, 'n_pixels': n_sqrt**2, 'n_sample_points': n_sqrt**2,
        'Z': 1,
        'obj_z_offset_min': 0.3,
        'projection_width': 0.0001,
    }
    build_params = util.param_table(build_param_values)
    run_params = util.param_table(run_param_values)
    print('build_params', build_param_values)
    print('run_params', run_param_values)

    results, results_fn = util.get_results(build, run, build_params, run_params, fn,
                                           n_trials=1, tmp_dir=DIR,
                                           copy_data=True, v=v)

    print('load')
    params, data = util.parse_file(DIR, f'{results_fn}.zip', 'out')

    # area per datapoint
    dS = params['z'][0]['width'] ** 2 / params['z'][0]['aspect_ratio']
    # squared amp / area
    z = data['z'][0][:, 0] ** 2 * dS
    z = z / z.max()  # normalized
    print('z', z.min())

    def loss(args):
        # args = mu1, mu2, s1, s2, s3, s4
        # z_pred = util.gaussian_1d(X, *args)
        z_pred = util.gaussian(X, *args)
        # Note that scaling biasses the model to large sigma
        z_pred = np.clip(z_pred / z_pred.max(), 1e-14, None)
        H = util.cross_entropy(z, z_pred)
        H += util.cross_entropy(z_pred, z)
        if np.isnan(H):
            return 1e3 * z.size
        assert H >= 0, H
        return H

    unit = 1e-6  # scale to improve fitting, required if sigma << 1
    X = data['w'][0][:, :2] / unit
    # result = scipy.optimize.minimize(loss, x0=(1, 0, 0, 0, 0,))
    result = scipy.optimize.minimize(loss, x0=(0, 0, 1))
    z_pred = util.gaussian(X, *result.x)
    # z_pred = util.gaussian_1d(X, 1)
    z_pred /= z_pred.max()
    X *= unit
    print(result.message)
    print('mu', result.x[:2] * unit)
    print('var', result.x[2] * unit)
    var = result.x[2] * unit

    for d in (2, 3):
        # cmap = 'OrRd'
        cmap = 'gist_heat'
        if d == 2:
            plt.figure(figsize=(8, 4))
            ax = plt.subplot(121)
            # plot._hist2d_wrapper(*(X.T * unit), z, bins=bins, cmap=cmap)
            plot._imshow_wrapper(*(X.T * unit), z.reshape((n_sqrt, n_sqrt)),
                                 cmap=cmap, vmin=0, vmax=1)
            # x, y = X.T * unit
            # plt.imshow(z.reshape((n_sqrt, n_sqrt)), origin='lower', aspect=1.,
            #            extent=(x.min(), x.max(), y.min(), y.max()),
            #            vmin=0, vmax=1)
            markup = {}
        else:
            plt.figure(figsize=(12, 6))
            ax = plt.subplot(121, projection='3d')
            surf.surf(*X.T, z, n_sqrt, n_sqrt, ax=ax)
            ax.set_zlim(0, 1)
            markup = {'colorbar': False, 'rotation': 0}
        # for surf colorbar see: https://stackoverflow.com/questions/6600579/colorbar-for-matplotlib-plot-surface-command
        plt.title('Irradiance')
        plot.markup(ax, unit='m', **markup)

        if d == 2:
            ax = plt.subplot(122)
            # plot._hist2d_wrapper(*(X.T * unit), z_pred, bins=bins, cmap=cmap)
            plot._imshow_wrapper(*(X.T * unit), z_pred.reshape((n_sqrt, n_sqrt)),
                                 cmap=cmap, vmin=0, vmax=1)
        else:
            ax = plt.subplot(122, projection='3d')
            surf.surf(*X.T, z_pred, n_sqrt, n_sqrt, ax=ax)
            ax.set_zlim(0, 1)
        mu = r'$\mathbf{\mu}=_0^0$'
        plt.title(f'Gaussian {mu}, $\\sigma^2={var:0.2e}$')
        plot.markup(ax, unit='m', **markup)
        # plt.suptitle('Irradiance')
        plt.tight_layout()
        plot.save_fig(f'{fn}-{d}d')


if __name__ == '__main__':
    v = 1
    n_trials = 2
    prefix = 'exp'
    # print('exp performance')
    # performance(n_trials, prefix + '-perf', v=0)
    # print('exp distribution object')
    # n = 128 ** 2
    # projection_width = n * 7e-6 * 0.1
    # distribution_vary_object(prefix + '-dist-obj', n, projection_width,  v)
    # print('exp distribution projector')
    # distribution_vary_projector(prefix + '-dist-proj', n, v)
    print('fit gaussian')
    fit_gaussian(prefix + '-gaus')
