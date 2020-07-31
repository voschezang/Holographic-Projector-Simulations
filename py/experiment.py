import numpy as np
import subprocess
import collections
import matplotlib.pyplot as plt
# local
import plot
import remote
import util

EXE = 'run_experiment'
CD = 'cd ../cuda'

translate = collections.defaultdict(lambda: 'value unknown')
translate.update({'Runtime mean': 'Runtime',
                  'FLOPS mean': 'FLOPS',
                  'kernel_size': 'Kernel size',
                  'n_objects': 'Objects',
                  })
translate_latex = translate.copy()
translate_latex['n'] = 'N'


def build(kernel_size=16):
    flags = '-l curand -l cublas -std=c++14'
    arch_flags = '-arch=compute_70 -code=sm_70'
    D = f"KERNEL_SIZE='{kernel_size}'"
    content = f"""
{remote.SSH} << EOF
source ~/.profile
cd cuda
nvcc -o {EXE} main.cu {flags} {arch_flags} -D{D}
EOF
"""
    return subprocess.run(content, shell=True, check=True, capture_output=True)


def run(n_objects=1, y=1920 * 1080, z=None, X=1, Z=0,
        obj_x_offset_min=0,
        obj_z_offset_min=1e-2, obj_z_offset_max=None,
        projection_z_offset_min=0., projection_z_offset_max=None,
        projection_width=None,
        aspect_ratio_projector=1., aspect_ratio_projection=1.,
        quadrant_projection=False, randomize_pixels=False):
    assert X >= 1
    if obj_z_offset_max is None:
        obj_z_offset_max = obj_z_offset_min
    print('obj_z_offset:', obj_z_offset_min, obj_z_offset_max)
    if projection_z_offset_max is None:
        projection_z_offset_max = projection_z_offset_min
    if z is None:
        z = y
    if projection_width is None:
        projection_width = z * 7e-6
    # projector_width = y * 7e-6 # TODO

    flags = [f'-x {n_objects} -y {y} -z {z} -X {X} -Z {Z}',
             f'-a {aspect_ratio_projector} -A {aspect_ratio_projection}',
             f'-u {obj_x_offset_min} -v 0',
             f'-w {obj_z_offset_min} -W {obj_z_offset_max}',
             f'-o 0. -O 0.',
             f'-n {projection_width}',
             f'-m {projection_z_offset_min} -M {projection_z_offset_max}',
             f'-q 1' * quadrant_projection,  # bool flag
             f'-r 1' * randomize_pixels,  # bool flag
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
    # print('content', content, '\n')
    return subprocess.run(content, shell=True, check=True, capture_output=True)


def performance(n_trials: int, fn: str, v=0):
    build_param_values = {'kernel_size': [8, 16]}
    run_param_values = {'n_objects': [8, 32, 128, 512]}
    build_params = util.param_table(build_param_values)
    run_params = util.param_table(run_param_values)
    print('build_params', build_param_values)
    print('run_params', run_param_values)

    results, _ = util.get_results(build, run, build_params, run_params, fn,
                                  n_trials=n_trials, v=v)
    # results = build_params.join(run_params).join(results)
    if v:
        print(results)
    for metric in ['Runtime', 'FLOPS']:
        for bar in [0, 1]:
            plot.grid_search_result(results, x_key='n_objects',
                                    y_key=f'{metric} mean',  z_keys=['kernel_size'],
                                    err_key=f'{metric} std', translate=translate,
                                    # ylog=metric == 'FLOPS',
                                    ylog=0,
                                    y_unit='s' if metric == 'Runtime' else None,
                                    bar=bar,
                                    interpolate=0,
                                    # interpolate=0 if bar else 'regression',
                                    err_alpha=0., v=v)
            # plt.ylim((0, None))
            if metric == 'FLOPS':
                # hlim = 1e11  # 7.4e12
                # plt.axhline(hlim, label='Theoretical limit', ls='--', lw=1,
                #             alpha=0.4, color='0')
                # plt.ylim(0, hlim * 1.01)
                # plt.legend()
                plt.title(f'Efficiency')
            else:
                plt.title(f'{metric}')
            plt.tight_layout()
            # plot.save(f'runtime-{bar}')
            plot.save_fig(f'{fn}-{metric}-{bar}', transparent=False)


def distribution_vary_object(fn: str, n: int, projection_width=1, v=0):
    build_param_values = {}
    run_param_values = {'obj_z_offset_min': 0.01,
                        'obj_z_offset_max': 1,
                        'projection_width': projection_width,
                        'aspect_ratio_projection': n,
                        'quadrant_projection': True,
                        'randomize_pixels': True,
                        'y': n, 'X': 4, 'Z': 1}
    build_params = util.param_table(build_param_values)
    run_params = util.param_table(run_param_values)
    print('build_params', build_param_values)
    print('run_params', run_param_values)

    dir = 'tmp_local'
    results, results_fn = util.get_results(build, run, build_params, run_params, fn,
                                           n_trials=1, tmp_dir=dir,
                                           copy_data=True, v=v)

    print('load')
    params, data = util.parse_file(dir, f'{results_fn}.zip', 'out')
    print('post')

    for xlog, ylog in np.ndindex((2, 2)):
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
                        'aspect_ratio_projection': n,
                        'quadrant_projection': True,
                        # 'randomize_pixels': True,
                        'y': n, 'X': 1, 'Z': m}
    build_params = util.param_table(build_param_values)
    run_params = util.param_table(run_param_values)
    print('build_params', build_param_values)
    print('run_params', run_param_values)

    dir = 'tmp_local'
    results, results_fn = util.get_results(build, run, build_params, run_params, fn,
                                           n_trials=1, tmp_dir=dir,
                                           copy_data=True, v=v)

    print('load')
    params, data = util.parse_file(dir, f'{results_fn}.zip', 'out')
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

    for xlog, ylog in np.ndindex((2, 2)):
        title = ('Log ' * ylog) + 'Amplitude Near Object'
        plot.distribution1d(data['w'], [z[:, 0] for z in data['z']], title,
                            xlog=xlog, ylog=ylog,
                            figshape=(2, 3), labels=labels)
        plot.save_fig(f'{fn}-amp-{xlog}{ylog}', transparent=False)


if __name__ == '__main__':
    v = 1
    n_trials = 2
    prefix = 'exp'
    # print('exp performance')
    # performance(n_trials, prefix + '-perf', v=0)
    print('exp distribution object')
    n = 128 ** 2
    projection_width = n * 7e-6 * 0.1
    distribution_vary_object(prefix + '-dist-obj', n, projection_width,  v)
    print('exp distribution projector')
    distribution_vary_projector(prefix + '-dist-proj', n, v)
