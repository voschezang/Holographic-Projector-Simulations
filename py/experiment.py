import collections
import itertools
import numpy as np
import pandas as pd
import subprocess
import scipy.optimize
import scipy.stats
import scipy.signal
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import matplotlib.image as mpimg


# local
import util
import remote
import plot
import surf

EXE = 'run_experiment'
CD = 'cd ../cuda'
DIR = 'tmp_local'
alpha_map = ('algorithm', {0: 0.5, 1: 0.6, 3: 0.8})
hatch_map = ('algorithm', {3: '//'})
# 'True MC': {True: '-', False: ''})

translate = collections.defaultdict(lambda: 'value unknown')
translate.update({'Runtime mean': 'Runtime',
                  'FLOPS mean': 'FLOPS',
                  'kernel_size': 'Kernel size',
                  'n_objects': 'N',
                  'n_pixels': 'M',
                  'n_pixelsN': 'N',  # alias with different translation
                  'n_sample_points': 'M',
                  'bin_size': 'Bin Size',
                  'N': r'$\sqrt{N\cdot M}$',
                  'n_streams': 'S',
                  'algorithm': '',
                  'thread_size': 'T',
                  'blockDim_x': 'B.x',
                  # 'blockDim_x': 'blockDim.x',
                  'blockDim_y': 'B.y',
                  'gridDim_x': 'G.x',
                  'gridDim_y': 'G.y',
                  'gridDim_y': 'gridDim.y',
                  # minor
                  'peak width mean': 'Peak Width',
                  'projector_width': 'Projector Width',
                  'projection_width': 'Projection Width',
                  'obj_z_offset_min': 'Offset',
                  'convergence_threshold': '$\epsilon$',
                  # 'true_MC': 'True MC'
                  # 'MC': 'Estimator'
                  'MC': '',
                  'MC_key': 'MC'
                  })
translate_latex = translate.copy()
translate_latex['n_objects'] = 'N'


def build(true_MC=False):
    flags = '-l curand -l cublas -std=c++14'
    arch_flags = '-arch=compute_70 -code=sm_70'
    macros = [f'RANDOMIZE_SUPERPOSITION_INPUT={true_MC:d}']
    # -D{' -D'.join(macros)}
    content = f"""
{remote.SSH} << EOF
source ~/.profile
cd cuda
nvcc -o {EXE} main.cu {flags} {arch_flags} -D{' -D'.join(macros)}
EOF
"""
    return subprocess.run(content, shell=True, check=True, capture_output=True)


def run(n_objects=1, n_pixels=1920 * 1080, n_sample_points=None, N=None,
        X=1, Z=0,
        n_pixelsN=None,
        obj_x_offset_min=0,
        obj_z_offset_min=0.35, obj_z_offset_max=None,
        projection_z_offset_min=0., projection_z_offset_max=None,
        projector_width=1920 * 7e-6,
        projection_width=None,
        aspect_ratio_projector=1., aspect_ratio_projection=1.,
        algorithm=1, quadrant_projection=False, randomize_pixels=False,
        n_streams=16, thread_size=(32, 32),
        blockDim_x=16, blockDim_y=16,
        gridDim_x=16, gridDim_y=16,
        convergence_threshold=0):
    assert N is not None or n_objects is not None
    assert X >= 1
    if Z is None:
        Z = 0
    if obj_z_offset_max is None:
        obj_z_offset_max = obj_z_offset_min
    if projection_z_offset_max is None:
        projection_z_offset_max = projection_z_offset_min
    if N is not None:
        n_objects = N
        n_pixels = N

    if n_pixelsN is not None:
        n_pixels = n_pixelsN
    elif n_pixels is None:
        n_pixels = n_objects

    if n_sample_points is None:
        n_sample_points = n_pixels
    if projection_width is None:
        projection_width = n_sample_points * 7e-6
    # projector_width = y * 7e-6 # TODO
    assert blockDim_x * blockDim_y <= 1024, 'Max number of threads per block'

    nonzero = [n_objects, n_pixels, n_sample_points, X,
               obj_z_offset_min, obj_z_offset_max,
               aspect_ratio_projector, aspect_ratio_projection,
               projector_width,
               projection_width,
               algorithm, n_streams,
               thread_size[0], thread_size[1],
               blockDim_x, blockDim_y,
               gridDim_x, gridDim_y]
    for i, v in enumerate(nonzero):
        assert v, f'Run param {i}:{v}'

    # TODO catch errors caused by "None" values in flags at program-runtime
    for i, v in enumerate(nonzero +
                          [Z, obj_x_offset_min,
                           projection_z_offset_min, projection_z_offset_max]):
        assert v is not None, f'Run param {i}:{v}'

    flags = [f'-x {n_objects} -y {n_pixels} -z {n_sample_points}',
             f'-X {X} -Z {Z}',
             f'-a {aspect_ratio_projector} -A {aspect_ratio_projection}',
             f'-u {obj_x_offset_min} -v 0',
             f'-w {obj_z_offset_min} -W {obj_z_offset_max}',
             f'-o 0.001',
             f'-l {projector_width}',
             f'-n {projection_width}',
             f'-m {projection_z_offset_min} -M {projection_z_offset_max}',
             f'-p {algorithm:d}',
             f'-q' * quadrant_projection,  # bool flag
             f'-r' * randomize_pixels,  # bool flag
             f'-s {n_streams}',
             f'-t {thread_size[0]} -T {thread_size[1]}',
             f'-b {blockDim_x} -B {blockDim_y}',
             f'-e {convergence_threshold}',
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
    # print('content', content, '\n')
    return subprocess.run(content, shell=True, check=True, capture_output=True)


def performance_matlab(fn: str, v=0):
    n_trials = 5
    # n_objects = np.array([256 ** 2, 1024 * 1080])
    results = None
    for run_param_values in [{'algorithm': 1,
                              'n_streams': [16],
                              'thread_size': [(32, 32)],
                              'blockDim_x': 8,
                              'blockDim_y': 8},
                             {'algorithm': [2, 3],
                              'n_streams': [32],
                              'thread_size': [(512, 256)],
                              'blockDim_x': 16,
                              'blockDim_y': 16
                              }]:
        N1 = 256 ** 2
        N2 = 1024 * 1080
        M1 = 1024 ** 2
        M2 = 1024 ** 2
        run_param_values['n_objects'] = N1
        run_param_values['n_pixels'] = M1
        run_params = util.param_table(run_param_values)
        results1, _ = util.get_results(build, run, None, run_params, fn,
                                       n_trials=n_trials, v=v)
        if results is None:
            results = results1
        else:
            results = results.append(results1, ignore_index=True, sort=False)
        # print(results.loc[:, ['algorithm', 'Runtime mean']])

        run_param_values['n_objects'] = N2
        run_param_values['n_pixels'] = M1
        run_params = util.param_table(run_param_values)
        results2, _ = util.get_results(build, run, None, run_params, fn,
                                       n_trials=n_trials, v=v)
        results = results.append(results2, ignore_index=True, sort=False)

    matlab1 = np.array([31.8891, 31.9071, 31.8971, 31.8803, 31.9318])
    matlab2 = np.array([995.1784, 998.1715, 999.0059, 998.7978, 998.2705])
    # add blockdim and thread size for compatibility with util.bandwidth
    results = results.append({'algorithm': 0, 'n_objects': N1, 'n_pixels': M1,
                              'Runtime mean': matlab1.mean(),
                              'Runtime std': matlab1.std(),
                              'FLOPS mean': (N1 * M1 * util.FLOP / matlab1).mean(),
                              'FLOPS std': (N1 * M1 * util.FLOP / matlab1).std(),
                              'n_streams': 1,
                              'thread_size': (1, 1),
                              'blockDim_x': 1
                              },
                             ignore_index=True, sort=False)
    results = results.append({'algorithm': 0, 'n_objects': N2, 'n_pixels': M2,
                              'Runtime mean': matlab2.mean(),
                              'Runtime std': matlab2.std(),
                              'FLOPS mean': (N2 * M2 * util.FLOP / matlab2).mean(),
                              'FLOPS std': (N2 * M2 * util.FLOP / matlab2).std(),
                              'n_streams': 1,
                              'thread_size': (1, 1),
                              'blockDim_x': 1
                              },
                             ignore_index=True, sort=False)
    # in GB/s
    bandwidth_args = (results.loc[:, 'algorithm'],
                      results.loc[:, 'n_objects'],
                      results.loc[:, 'n_pixels'],
                      results.loc[:, 'Runtime mean'],
                      results.loc[:, 'n_streams'],
                      results.loc[:, 'thread_size'],
                      results.loc[:, 'blockDim_x'],
                      16)
    results.loc[:, 'Effective Bandwidth (UB)'] = util.bandwidth(
        *bandwidth_args, cache_threads=0, cache_blocks=0)
    results.loc[:, 'Effective Bandwidth (LB)'] = util.bandwidth(
        *bandwidth_args, cache_threads=1, cache_blocks=0)
    results.loc[:, 'Effective Bandwidth (LB g)'] = util.bandwidth(
        *bandwidth_args, cache_threads=1, cache_blocks=1)

    results.loc[:, 'b2'] = 1e-9 * 8 * (
        1 * results.loc[:, 'n_objects'].values * results.loc[:, 'n_pixels'].values +
        5 * results.loc[:, 'n_objects'].values +
        # 3 * results.loc[:, 'n_pixels'].values +
        2 * results.loc[:, 'n_pixels'].values
    ) / \
        results.loc[:, 'Runtime mean'].values

    # print(results.loc[:, 'b2'])
    # print(results.loc[:, 'n_pixels'].values /
    #       results.loc[:, 'Runtime mean'].values)
    # print(results.loc[:, 'Effective Bandwidth'])
    results = results.loc[:, ['algorithm', 'n_objects', 'n_pixels',
                              'Runtime mean', 'Runtime std',
                              'FLOPS mean', 'FLOPS std',
                              # 'b2',
                              # 'Effective Bandwidth',
                              'Effective Bandwidth (UB)',
                              'Effective Bandwidth (LB)',
                              'Effective Bandwidth (LB g)',
                              ]]
    print(results)
    print('baseline', matlab1.mean(), matlab2.mean())
    print('> Min. runtime: %f s\n>Max. performance: %f TFLOPS' %
          (results.loc[:, 'Runtime mean'].min(),
           results.loc[:, 'FLOPS mean'].max() * 1e-12))
    for i, metric in enumerate(['Runtime', 'FLOPS', 'Runtime']):
        y_key = f'{metric} mean'
        maxi = results.loc[:, y_key].max()
        # for i, n in enumerate([N1, N2]):
        n = 0
        if 1:
            selection = results
            if i == 2:
                # if metric == 'Runtime':
                selection = results.query(f'algorithm > 0')
            if selection.shape[0] == 0:
                print('No results for params', filter)
                continue

            # fig = plt.figure(figsize=(7, 4))
            # fig = plt.figure(figsize=(4, 4))
            if metric == 'FLOPS':
                fig = plt.figure(figsize=(4, 3))
            else:
                fig = plt.figure(figsize=(3, 3))
            plot.grid_search_result(selection, x_key='n_objects',
                                    y_key=y_key,
                                    z_keys=[
                                        'algorithm',
                                    ],
                                    err_key=f'{metric} std' if i < 2 else None,
                                    alpha_map=('algorithm',
                                               {0: 0.5, 1: 0.6, 3: 0.8}),
                                    hatch_map=('algorithm', {
                                        # 1: '\\',
                                        3: '//'}),
                                    translate=translate,
                                    ylog=0,
                                    baseline=[matlab1.mean(), matlab2.mean()] \
                                    if i == 2 else None,
                                    bar=1,
                                    interpolate=0,
                                    # loc='upper left',
                                    legend=metric == 'FLOPS',
                                    sort_keys=False,
                                    fig=fig,
                                    err_alpha=0., v=v)
            if metric == 'FLOPS':
                plt.title(f'Efficiency')
                plt.ylim(0, maxi * 1.05)
                plt.legend(loc='upper left', bbox_to_anchor=(1., 1))
            elif i == 2:
                plt.title('Speedup')
                plt.ylabel('Speedup')
                plt.ylim(0, 50)

            plt.tight_layout()
            # plot.save(f'runtime-{bar}')
            plot.save_fig(f'{fn}-{metric}-{n}-{i}', transparent=False)

    plt.close()

    selection = results.query(f'algorithm > 0 and n_objects == {N2}')
    xticks = np.arange(selection.shape[0])
    w = 0.35
    plt.bar(xticks - w / 2,
            selection.loc[:, 'Effective Bandwidth (UB)'], w, label='LB',
            zorder=2)
    plt.bar(xticks + w / 2,
            selection.loc[:, 'Effective Bandwidth (LB)'], w, label='UB',
            zorder=2)
    ax = plt.gca()
    plot.sci_labels(ax)
    plt.xticks(xticks, labels=[f'#{k}' for k in selection.loc[:, 'algorithm']],
               rotation=0)
    # plot.markup(ax, unit='B/s')
    ax.yaxis.set_major_formatter(tck.EngFormatter(places=1, unit='B/s'))
    plt.ylabel('Bandwidth')
    plt.xlabel('Kernel')
    # plt.axhline(870e9, color='0', ls=':', zorder=2,
    #             label='Theoretical Bandwidth')
    # plt.legend(loc='upper right')
    # plt.legend()
    plt.grid(which='major', linewidth=0.4, axis='y')
    plt.grid(which='minor', linewidth=0.2, axis='y')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # plt.yscale('log')
    plt.tight_layout()
    plot.save_fig(f'bandwidth', transparent=False)
    plt.close()

    # for i, y_key in enumerate(['Effective Bandwidth (UB)',
    #                            'Effective Bandwidth (LB)',
    #                            # 'Effective Bandwidth (LB g)'
    #                            ]):
    #     fig = plt.figure(figsize=(8, 4))
    #     plot.grid_search_result(selection, x_key='algorithm',
    #                             y_key=y_key,
    #                             # z_keys=['algorithm'],
    #                             translate=translate, bar=1, sort_keys=False,
    #                             # loc='upper right',
    #                             fig=fig, v=0)
    #     ax = plt.gca()
    #     formatter = tck.EngFormatter(
    #         places=1, sep=u"\N{THIN SPACE}", unit='B/s')
    #     ax.yaxis.set_major_formatter(formatter)
    #
    #     plt.title(y_key)
    #     plt.ylabel('Bandwidth')
    #     # plt.axhline(870e9, color='0', ls=':', zorder=2,
    #     #             label='Theoretical Bandwidth')
    #     plt.legend(loc='upper right')
    #     plt.tight_layout()
    # plot.save_fig(f'bandwidth-{i}', transparent=False)
    # plt.close()


def performance(n_trials: int, fn: str, bar=1, v=0):
    n_pixels = np.array([1e4, 1e5, 1e6, 1e7])
    # n_pixels = np.array([256, 512, 1024, 2048]) * 1024
    # n_pixels = np.array([256, 512, 1024, 2048]) ** 2 # old
    run_param_values = {
        'n_objects': [256 ** 2],
        'n_pixels': n_pixels,
        'algorithm': [1, 2, 3],
        'n_streams': [16],
        'thread_size': [(32, 32)],
        # 'thread_size': [(32, 16), (32, 32)],
        # 'blockDim_x': [8, 16],
        'blockDim_x': [8, 32],
        'blockDim_y': [8],
    }
    run_params = util.param_table(run_param_values)
    print('run_params', run_param_values)

    results, results_fn = util.get_results(build, run, None, run_params, fn,
                                           n_trials=n_trials, v=v)
    print('> Min. runtime: %f s\n>Max. performance: %f TFLOPS' %
          (results.loc[:, 'Runtime mean'].min(),
           results.loc[:, 'FLOPS mean'].max() * 1e-12))
    print(results.keys())
    for m in n_pixels:
        print(f"M: {np.sqrt(m).astype(int)}^2")
        for a in [1, 2, 3]:
            min = results.query(
                f"algorithm == {a} and n_pixels == {m}").loc[:, 'Runtime mean'].min()
            # (f"'Runtime mean' == {results.loc[:, 'Runtime mean'].min()}")
            # print('a', results.query(f"algorithm == {a}"))
            # print('m', results.query(f"`Runtime mean` == {min}"))
            print(results.query(f"algorithm == {a} and `Runtime mean` == {min}")
                  .loc[:, ['n_pixels', 'algorithm', 'Runtime mean', 'Runtime std']])
    blockdims = run_param_values['blockDim_x']
    for metric in ['Runtime', 'FLOPS']:
        y_key = f'{metric} mean'
        maxi = results.loc[:, y_key].max()
        for g, b in [(0, 0)]:
            selection = results
            if selection.shape[0] == 0:
                print('No results for params', filter)
                continue
            # fig = plt.figure(figsize=(7, 4))
            fig = plt.figure(figsize=(8, 3))
            plot.grid_search_result(selection, x_key='n_pixels',
                                    y_key=y_key,
                                    z_keys=[
                                        'algorithm',
                                        # 'thread_size',
                                        'blockDim_x',
                                    ],
                                    alpha_map=('blockDim_x',
                                               util.make_map(blockdims)),
                                    color_group_size=len(blockdims),
                                    err_key=f'{metric} std',
                                    # alpha_map=('algorithm', {1: 0.6, 3: 0.8}),
                                    hatch_map=('algorithm', {
                                               # 1: '\\',
                                               3: '//'}),
                                    translate=translate,
                                    ylog=0,
                                    y_unit='s' if metric == 'Runtime' else None,
                                    bar=bar,
                                    interpolate=0,
                                    # loc='upper left',
                                    # legend=metric == 'Runtime',
                                    fig=fig,
                                    err_alpha=0., v=v)
            # suffix = ''
            # if metric == 'FLOPS':
            #     plt.title(f'Efficiency{suffix}')
            # else:
            #     plt.title(f'{metric}{suffix}')
            plt.ylim(0, maxi * 1.05)
            # plt.xscale('log')
            plt.tight_layout()
            # plot.save(f'runtime-{bar}')
            plot.save_fig(f'{fn}-{metric}-{g}-{b}-c', transparent=False)

    plt.close()


def performance1(n_trials: int, fn: str,
                 xlog=0, v=0,
                 algorithm=2, n_streams=8,
                 n_objects_options=[[64]],
                 **run_param_values):
    run_param_values['algorithm'] = algorithm
    run_param_values['n_streams'] = n_streams
    results = None
    for i, n_objects in enumerate(n_objects_options):
        run_param_values['n_objects'] = n_objects
        if i > 0:
            run_param_values['algorithm'] = [2, 3]
        run_params = util.param_table(run_param_values)
        results1, results_fn = util.get_results(build, run, None, run_params,
                                                fn, n_trials=n_trials, v=v)
        print('> Min. runtime: %f s\n>Max. performance: %f TFLOPS' %
              (results1.loc[:, 'Runtime mean'].min(),
               results1.loc[:, 'FLOPS mean'].max() * 1e-12))
        if results is None:
            results = results1
        else:
            results = results.append(results1, ignore_index=True, sort=False)
    thread_sizes = run_param_values['thread_size']
    for metric in ['Runtime', 'FLOPS']:
        y_key = f'{metric} mean'
        maxi = results.loc[:, y_key].max()
        selection = results
        # selection = results.query(f'n_objects > 2e4')
        if selection.shape[0] == 0:
            print('No results for params', filter)
            continue
        for bar in [0]:
            # fig = plt.figure(figsize=(8, 5))
            fig = plt.figure(figsize=(9, 3))
            plot.grid_search_result(selection, x_key='n_objects',
                                    y_key=y_key,
                                    z_keys=[
                                        'algorithm',
                                        # 'thread_size',
                                    ],
                                    alpha_map=('thread_size',
                                               util.make_map(thread_sizes)),
                                    color_group_size=len(thread_sizes),
                                    err_key=f'{metric} std',
                                    translate=translate,
                                    ylog=0,
                                    y_unit='s' if metric == 'Runtime' else None,
                                    x_func=lambda x: x if bar else x * 1e-6,
                                    bar=bar,
                                    interpolate=0 if bar or metric == 'FLOPS' else 'regression',
                                    fig=fig,
                                    # scatter_markers=itertools.cycle('x+'),
                                    err_alpha=0., v=v)
            plt.ylim(0, maxi * 1.05)
            plt.title('Efficiency' if metric == 'FLOPS' else metric,
                      fontsize=14)
            # ax = plt.gca()
            # _, labels = ax.get_legend_handles_labels()
            # Note, the length will depend on the latex formatting
            # plt.legend(loc='upper left', bbox_to_anchor=(1., 1),
            #            labels=[f'{label:<16}' for label in labels])
            if xlog:
                plt.xscale('log')
            if not bar:
                ax = plt.gca()
                ax.xaxis.set_major_locator(tck.AutoLocator())
                # formatter = tck.EngFormatter(places=decimals, sep=u"\N{THIN SPACE}",
                #                  unit=unit)
                # ax.xaxis.set_major_formatter(tck.)
                plt.xlabel(r'N ($\times$ 10$^6$)')
            plt.tight_layout()
            # plot.save(f'runtime-{bar}')
            plot.save_fig(f'{fn}-{metric}-{bar}', transparent=False)

    plt.close()


def performance1b(n_trials: int, fn: str,
                  xlog=0, v=0, algorithm=2, n_streams=8,
                  nm_options=[([64], [64, 128])],
                  interpolate='regression',
                  interpolate_lb=0,
                  x_key='n_pixels',
                  z_keys=['algorithm', 'n_objects'],
                  alpha_key='thread_size',
                  scale_x=1e-6,
                  figsize=(9, 3),
                  legend=True,
                  move_legend=False,
                  **run_param_values):
    # scaling of m, for different n
    run_param_values['algorithm'] = algorithm
    run_param_values['n_streams'] = n_streams
    results = None
    max_n_pixels = 0
    for i, (n_objects, n_pixels) in enumerate(nm_options):
        run_param_values['n_objects'] = n_objects
        run_param_values['n_pixels'] = n_pixels
        run_params = util.param_table(run_param_values)
        max_n_pixels = max(max_n_pixels, max(n_pixels))
        results1, results_fn = util.get_results(build, run, None, run_params,
                                                fn, n_trials=n_trials, v=v)
        print('> Min. runtime: %f s\n>Max. performance: %f TFLOPS' %
              (results1.loc[:, 'Runtime mean'].min(),
               results1.loc[:, 'FLOPS mean'].max() * 1e-12))
        if results is None:
            results = results1
        else:
            results = results.append(results1, ignore_index=True, sort=False)
    thread_sizes = run_param_values['thread_size']
    alpha_map = (alpha_key, util.make_map(run_param_values[alpha_key]))
    for metric in ['Runtime', 'FLOPS']:
        y_key = f'{metric} mean'
        maxi = results.loc[:, y_key].max()
        selection = results
        # selection = results.query(f'n_objects > 2e4')
        if selection.shape[0] == 0:
            print('No results for params', filter)
            continue
        for bar in [0, 1]:
            def scale(x): return x if bar else x * scale_x
            # interp_subplot = 0 if bar or metric == 'Runtime' else interpolate
            # interp_subplot = 0 if bar or metric == 'FLOPS' else interpolate
            interp_subplot = 0 if bar else interpolate
            # fig = plt.figure(figsize=(8, 5))
            fig = plt.figure(figsize=figsize)
            plot.grid_search_result(selection, x_key=x_key,
                                    y_key=y_key,
                                    z_keys=z_keys,
                                    # alpha_map=alpha_map,
                                    color_group_size=len(thread_sizes),
                                    err_key=f'{metric} std',
                                    translate=translate,
                                    ylog=0,
                                    y_unit='s' if metric == 'Runtime' else None,
                                    bar=bar,
                                    x_func=scale,
                                    interpolate=interp_subplot,
                                    # or metric == 'FLOPS' else interpolate
                                    interpolate_lb=scale(interpolate_lb),
                                    legend=legend,
                                    fig=fig,
                                    err_alpha=0., v=v)
            if interp_subplot:
                plt.axvline(scale(interpolate_lb), ls='-', lw=1.8,
                            color='tab:red', alpha=0.3)
            plt.ylim(0, maxi * 1.07)
            plt.title('Efficiency' if metric == 'FLOPS' else metric,
                      fontsize=14)
            # if legend:
            #     if metric == 'Runtime':
            #         if figsize[1] >= 3:
            #             plt.legend(loc='upper left', bbox_to_anchor=(1., 1))
            #         else:
            #             plt.legend(loc='upper left', bbox_to_anchor=(1., 1.2))
            if move_legend:
                plt.legend(loc='upper left', bbox_to_anchor=(1., 1))
            if xlog:
                plt.xscale('log')
            if not bar:
                ax = plt.gca()
                ax.xaxis.set_major_locator(tck.AutoLocator())
                plt.xlabel(translate[x_key] + r' ($\times$ 10$^6$)')
                plt.xlim(0, scale(max_n_pixels) * 1.05)
            plt.tight_layout()
            # plot.save(f'runtime-{bar}')
            plot.save_fig(f'{fn}-{metric}-{bar}', transparent=False)

    plt.close()


def performance2(n_trials: int, fn: str,
                 bar=1, xlog=0, v=0, **run_param_values):
    run_param_values['algorithm'] = [2, 3]
    run_param_values['n_streams'] = 32
    run_params = util.param_table(run_param_values)
    print('run_params', run_param_values)

    results, results_fn = util.get_results(build, run, None, run_params, fn,
                                           n_trials=n_trials, v=v)
    print('> Min. runtime: %f s\n>Max. performance: %f TFLOPS' %
          (results.loc[:, 'Runtime mean'].min(),
           results.loc[:, 'FLOPS mean'].max() * 1e-12))
    thread_sizes = run_param_values['thread_size']
    for metric in ['Runtime', 'FLOPS']:
        y_key = f'{metric} mean'
        maxi = results.loc[:, y_key].max()
        # for g, b in itertools.product([8, 16], [8, 16]):
        for g, b in [(0, 0)]:
            selection = results
            # selection = results.query(f'gridDim_x == {g}')
            filter = f'blockDim_x == {b} and gridDim_x == {g}'
            print(filter)
            # selection = results.query(filter)
            if selection.shape[0] == 0:
                print('No results for params', filter)
                continue
            # fig = plt.figure(figsize=(8, 5))
            fig = plt.figure(figsize=(8, 3))
            plot.grid_search_result(selection, x_key='n_objects',
                                    y_key=y_key,
                                    z_keys=[
                                        'algorithm',
                                        'thread_size',
                                    ],
                                    alpha_map=('thread_size',
                                               util.make_map(thread_sizes)),
                                    color_group_size=len(thread_sizes),
                                    # alpha_map=('algorithm', {3: 0.8}),
                                    hatch_map=('algorithm', {3: '//'}),
                                    err_key=f'{metric} std',
                                    translate=translate,
                                    ylog=0,
                                    y_unit='s' if metric == 'Runtime' else None,
                                    bar=bar,
                                    interpolate=0,
                                    fig=fig,
                                    err_alpha=0., v=v)
            # plt.ylim((0, None))
            # suffix = ''
            # if metric == 'FLOPS':
            #     plt.title(f'Efficiency{suffix}')
            # else:
            #     plt.title(f'{metric}{suffix}')
            plt.ylim(0, maxi * 1.05)
            plt.tight_layout()
            # plot.save(f'runtime-{bar}')
            plot.save_fig(f'{fn}-{metric}-{g}-{b}', transparent=False)

    plt.close()


def performance3(n_trials: int, fn: str,
                 blockDim_x=16, blockDim_y=16,
                 gridDim_x=16, gridDim_y=16,
                 v=0, bar=1, **run_param_values):
    run_param_values['blockDim_x'] = blockDim_x
    run_param_values['blockDim_y'] = blockDim_y
    run_param_values['gridDim_x'] = gridDim_x
    run_param_values['gridDim_y'] = gridDim_y
    run_param_values['algorithm'] = [2, 3]
    run_param_values['n_streams'] = 32
    run_params = util.param_table(run_param_values)
    print('run_params', run_param_values, '\n')

    results, results_fn = util.get_results(build, run, None, run_params, fn,
                                           n_trials=n_trials, v=v)
    print('> Min. runtime: %f s\n>Max. performance: %f TFLOPS' %
          (results.loc[:, 'Runtime mean'].min(),
           results.loc[:, 'FLOPS mean'].max() * 1e-12))
    print('n obj:', run_param_values['n_objects'])
    for metric in ['Runtime', 'FLOPS']:
        y_key = f'{metric} mean'
        maxi = results.loc[:, y_key].max()
        # b = 0
        # for g, b in itertools.product([8, 16], [8, 16]):
        # for g in [8, 16]:
        # for g, b in [(0, 0)]:
        for i, n in enumerate(run_param_values['n_objects']):
            selection = results
            # selection = results.query(f'gridDim_x == {g}')
            # filter = f'gridDim_y == {g}'
            filter = f'n_objects == {n}'
            print(filter)
            selection = results.query(filter)
            if selection.shape[0] == 0:
                print('No results for params', filter)
                continue
            # fig = plt.figure(figsize=(8, 4))
            fig = plt.figure(figsize=(9, 3))
            plot.grid_search_result(selection,
                                    # x_key='n_objects',
                                    x_key='gridDim_y',
                                    y_key=y_key,
                                    z_keys=[
                                        'algorithm',
                                        # 'thread_size',
                                        # 'blockDim_x',
                                        'blockDim_y',
                                        # 'gridDim_x',
                                        # 'gridDim_y',
                                    ],
                                    alpha_map=('algorithm', {3: 0.8}),
                                    hatch_map=('algorithm', {3: '//'}),
                                    err_key=f'{metric} std',
                                    translate=translate,
                                    ylog=0,
                                    y_unit='s' if metric == 'Runtime' else None,
                                    bar=bar,
                                    interpolate=0,
                                    loc='lower left',
                                    format_xlabels=0,
                                    fig=fig,
                                    err_alpha=0., v=v)
            # plt.ylim((0, None))
            suffix = ''
            # suffix = f'\n(Algorithm {a})'
            if metric == 'FLOPS':
                # hlim = 1e11  # 7.4e12
                # plt.axhline(hlim, label='Theoretical limit', ls='--', lw=1,
                #             alpha=0.4, color='0')
                # plt.ylim(0, hlim * 1.01)
                # plt.legend()
                plt.title(f'Efficiency{suffix}')
            else:
                plt.title(f'{metric}{suffix}')
            plt.legend(loc='upper left', bbox_to_anchor=(1., 1))
            plt.ylim(0, maxi * 1.05)
            plt.tight_layout()
            # plot.save(f'runtime-{bar}')
            plot.save_fig(f'{fn}-{metric}-{i}', transparent=False)

    plt.close()


def performance_cMC(n_trials: int, fn: str, n_streams=8,
                    # convergence_threshold= [1e-4, 5e-3, 1e-3, 1e-2],
                    convergence_threshold=[1e-5, 1e-4, 1e-3, 1e-2],
                    bin_sizes=np.arange(4, 36 + 1, 2) ** 2,
                    z_keys=['convergence_threshold'],
                    color_group_size=1,
                    xlim=None,
                    ylim=None,
                    v=0):
    # cond mc
    # TODO manually enable conditional_MC2 flag in cuda
    # n_trials = 1
    Z = n_trials
    # use multiple obj locations to increase variance between trials

    # N must be divisible by grid_size_x
    # grid_size_x = 8 * 8 * 4  # 256
    tx, ty = 128, 4
    tx, ty = 32, 4
    Bx, Gx = 4, 8
    grid_size_x = Gx * Bx * tx  # 256
    grid_size_y = 8 * 8 * ty
    assert util.is_square(grid_size_y)
    M = grid_size_y * 64
    # bin_sizes = np.array([4, 9, 16, 25, 64, 256, 1024, 4096][1:])  # squares
    # bin_sizes = np.arange(4, 32 + 1, 4) ** 2
    # bin_sizes = np.arange(4, 16 + 1, 2) ** 2
    # bin_sizes = np.arange(4, 48 + 1, 2) ** 2
    # N = [x for x in grid_size_x * bin_sizes if int(np.sqrt(x)) ** 2 == x]
    # N = bin_sizes * grid_size_x
    N = (np.sqrt(grid_size_x) * np.sqrt(bin_sizes))**2
    # print('N values pre ', N)
    # N = np.array([x for x in N if util.is_square(x)])
    print('grid_size_x', grid_size_x)
    print('N values', N)
    print('sqrt N', N ** .5)
    print('bin sizes', bin_sizes)
    print('bin sizes', (N ** .5 / grid_size_x ** .5) ** 2)
    print('sqrt bin sizes', N ** .5 / grid_size_x ** .5)
    print(N / grid_size_x)
    print('M', M, 'grid_size_y', grid_size_y)
    print('grid_size_x', grid_size_x)
    print('x data per stream', N / 8)
    # N2 = [x for x in 4 * grid_size_x * bin_sizes if int(np.sqrt(x)) ** 2 == x]
    # N = [x for x in N if x in N2]
    run_param_values = {
        'Z': Z,
        # 'convergence_threshold': [-1],  # -1 for no convergence
        # -1 for no convergence
        'obj_z_offset_min': 0.02,
        'obj_z_offset_max': 0.5,
        # 'convergence_threshold': [1e-16, 1e-5, 1e-4, 1e-3, 1e-2],
        # 'convergence_threshold': [1e-5, 5e-3, 1e-3, 1e-2],
        # 'convergence_threshold': [1e-4, 5e-3, 1e-3, 1e-2],
        'convergence_threshold': convergence_threshold,
        'n_objects': 64,
        # 'n_pixelsN': np.array([64, 128, 256, 512, 1024]) ** 2,
        # 'n_pixelsN': (np.arange(1, 5) * 512) ** 2,
        # 'n_pixelsN': (np.arange(1, 6) * 512) ** 2,
        # 'n_pixelsN': (np.arange(1, 8) * 256) ** 2,
        'n_pixelsN': N,
        # 'n_pixelsN': np.arange(4, 32 + 1, 4) ** 4,
        # bin size must be a square, as well as total N
        'n_sample_points': 256 ** 2,  # = 16^4
        'algorithm': 2,
        'n_streams': n_streams,
        # 'thread_size': [(4, 16), (4, 32), (16, 16)],
        # 'thread_size': [(256, 8)],
        'thread_size': [(tx, 16)],
        'blockDim_x': [Bx], 'blockDim_y': [8],
        'gridDim_x': [Gx], 'gridDim_y': [8]}
    run_params = util.param_table(run_param_values)
    results, results_fn = util.get_results(build, run, None, run_params, fn,
                                           n_trials=n_trials, v=v)

    results['bin_size'] = results.loc[:, 'n_pixelsN'] / grid_size_x
    print(run_param_values['convergence_threshold'])
    alpha_map = util.make_map(run_param_values['convergence_threshold'],
                              # [0.65] * 3 + [1] * 2
                              )
    global translate
    translate = translate.copy()
    translate['MC_key'] = ''
    # selection = results.query('n_streams == 16')
    # print('significant slopes (ignore last => poor fit):',
    #       {1e-5: 4e-6, 1e-4: 3.9e-6, 1e-3: -1.4e-6})
    for metric in ['Runtime', 'FLOPS']:
        y_key = f'{metric} mean'
        maxi = results.loc[:, y_key].max()
        maxi += results.loc[:, f'{metric} std'].max()
        # for a in [1, 2, 3]:
        # for s in [8, 16, 32]:
        for bar in [0, 1]:
            selection = results
            # selection = results.query(f'convergence_threshold != 1e-5')
            # print(selection.loc[:, ['n_pixelsN', 'thread_size',
            #                         'convergence_threshold', 'Runtime mean']])
            if selection.shape[0] == 0:
                continue
            # fig = plt.figure(figsize=(5, 6))
            fig = plt.figure(figsize=(9, 3))
            plot.grid_search_result(selection, x_key='n_pixelsN',  # bin_size n_pixelsN
                                    y_key=y_key,
                                    z_keys=z_keys,
                                    err_key=f'{metric} std',
                                    alpha_map=('convergence_threshold',
                                               alpha_map),
                                    translate=translate,
                                    ylog=0,
                                    y_unit='s' if metric == 'Runtime' else None,
                                    bar=bar,
                                    interpolate=0 if bar else 'pos-regression',
                                    omit_zero=False,
                                    x_func=lambda x: x if bar else x * 1e-6,
                                    fig=fig, color_group_size=color_group_size,
                                    err_alpha=0., v=v)
            if ylim:
                plt.ylim(ylim)
            else:
                plt.ylim(0, maxi * 1.05)
            # plt.title(f'{metric} ({s} streams)')
            if not bar:
                plt.xlabel(r'N ($\times$ 10$^6$)')
                # plt.xlabel(r'N ($\times$ 1,000,000)')
                if xlim:
                    plt.xlim(xlim)
                else:
                    plt.xlim(0)
            # if bar:
            #     plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1))
            # else:
            plt.legend(loc='upper left', bbox_to_anchor=(1., 1))
            plt.tight_layout()
            plot.save_fig(f'{fn}-{metric}-{bar}', transparent=False)
    plt.close()


def performance_MC3(n_trials: int, fn: str, v=0):
    # uncond MC
    # TODO manually disable conditional_MC2 flag in cuda
    ty = 4
    grid_size_y = 8 * 8 * ty
    assert util.is_square(grid_size_y)
    M = grid_size_y * 64
    # best performance: t: (64, 8)
    thread_sizes = [(64, ty), (256, ty)]  # first is better
    # thread_sizes = [(64, 8), (256, 64)]  # first is better
    # thread_sizes = [(32, 8), (64, 64), (256, 8)]  # first is better
    # thread_sizes = [(32, 32), (64, 32)]  # first is better
    run_param_values = {
        'Z': 1,
        'convergence_threshold': [1e-16],  # .000001 for no convergence
        'n_objects': 64,  # non-trivial distribution
        # 'n_pixelsN': np.array([256, 512]) ** 2,
        # 'n_pixelsN': np.array([512, 1024, 2024]) ** 2,
        # 'n_pixelsN': (np.arange(1, 4) * 256)**2,
        'n_pixelsN': (np.arange(1, 5) * 512)**2,
        # 'n_sample_points': 4 * 1024,
        'n_sample_points': M,
        'algorithm': 2,
        'n_streams': 8,
        # 'thread_size': [(1, 32), (32, 1), (16, 16)],
        # 'thread_size': [(64, 8)],
        'thread_size': thread_sizes,
        'blockDim_x': [16], 'blockDim_y': [16],
        'gridDim_x': [8], 'gridDim_y': [8]}
    build_params = util.param_table({'true_MC': True})
    run_params = util.param_table(run_param_values)
    results, results_fn = util.get_results(build, run, build_params, run_params, fn,
                                           n_trials=n_trials, v=v)
    build_params = util.param_table({'true_MC': False})
    run_param_values['convergence_threshold'] += [0]
    run_params = util.param_table(run_param_values)
    results2, results_fn2 = util.get_results(build, run, build_params, run_params, fn,
                                             n_trials=n_trials, v=v)
    results = results.append(results2, ignore_index=True, sort=False)
    # print(results.loc[:, ['true_MC']])

    # selection = results.query('n_streams == 16')
    for metric in ['Runtime', 'FLOPS']:
        y_key = f'{metric} mean'
        maxi = results.loc[:, y_key].max()
        maxi += results.loc[:, f'{metric} std'].max()
        # print(metric, maxi)
        # for a in [1, 2, 3]:
        selection = results
        fig = plt.figure(figsize=(8, 3))
        # fig = plt.figure(figsize=(10, 3))
        bar = 0
        plot.grid_search_result(selection, x_key='n_pixelsN',
                                y_key=y_key,
                                z_keys=[
                                    'true_MC',
                                    'convergence_threshold',  # required for parsing
                                    'thread_size',
                                ],
                                err_key=f'{metric} std',
                                translate=translate,
                                alpha_map=('thread_size',
                                           util.make_map(thread_sizes)),
                                color_group_size=len(thread_sizes),
                                # ylog=metric == 'FLOPS',
                                ylog=0,
                                y_unit='s' if metric == 'Runtime' else None,
                                bar=bar,
                                loc='upper left',
                                fig=fig,
                                x_func=lambda x: x if bar else x * 1e-6,
                                # sort_keys=False,
                                interpolate=0 if bar else 'regression',
                                err_alpha=0., v=v)
        # plt.ylim((0, None))
        plt.xlabel(r'N ($\times$ 10$^6$)')
        plt.ylim(0, maxi * 1.05)
        plt.xlim(left=0)
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.tight_layout()
        # plot.save(f'runtime-{bar}')
        plot.save_fig(f'{fn}-{metric}-1', transparent=False)
    plt.close()


def performance_MC4(n_trials: int, fn: str, v=0):
    # uncond MC
    # TODO manually disable conditional_MC2 flag in cuda

    tx, ty = 256, 4
    # grid_size_x = 8 * 4 * tx  # 256
    grid_size_y = 8 * 8 * ty
    assert util.is_square(grid_size_y)
    M = grid_size_y * 256
    convergence_thresholds = [1e-16, 1e-4, 1e-3, 1e-2]
    global translate
    translate = translate.copy()
    translate['MC_key'] = ''
    run_param_values = {
        'Z': 1,
        # 'convergence_threshold': [1e-16],  # .000001 for no convergence
        'convergence_threshold': convergence_thresholds,  # .000001 for no convergence
        # 'convergence_threshold': [1e-3],  # -1 for no convergence
        'n_objects': 64,
        # 'n_pixelsN': 512**2,
        'n_pixelsN': np.array([256, 512, 1024])**2,
        # 'n_pixelsN': (np.arange(1, 4) * 512)**2,
        # 'n_sample_points': 512 ** 2,
        'n_sample_points': M,
        # 'n_sample_points': np.arange(3) * 1024,
        'algorithm': 2,
        'n_streams': 8,
        # 'thread_size': [(64, 8)],
        'thread_size': [(tx, 8)],
        'blockDim_x': [16],
        'blockDim_y': [16],
        'gridDim_x': [8],
        'gridDim_y': [8]}
    print(run_param_values['n_pixelsN'])
    build_params = util.param_table({'true_MC': True})
    run_params = util.param_table(run_param_values)
    results, results_fn = util.get_results(build, run, build_params, run_params, fn,
                                           n_trials=n_trials, v=v)
    build_params = util.param_table({'true_MC': False})
    # run_param_values['convergence_threshold'] += [0]
    run_params = util.param_table(run_param_values)
    results2, results_fn2 = util.get_results(build, run, build_params, run_params, fn,
                                             n_trials=n_trials, v=v)
    results = results.append(results2, ignore_index=True, sort=False)

    # print(results.loc[:, ['true_MC', 'Runtime mean', 'n_pixelsN']])
    # selection = results.query('n_streams == 16')
    for metric in ['Runtime', 'FLOPS']:
        y_key = f'{metric} mean'
        # maxi = results.loc[:, y_key].max()
        # maxi += results.loc[:, f'{metric} std'].max()
        # print(metric, maxi)
        # for i, n in enumerate(run_param_values['n_pixelsN']):
        for i, n in enumerate([True, False]):
            # legend = i == 2
            legend = i == 0
            # selection = results
            # selection = results.query(f'n_pixelsN == {n}')
            selection = results.query(f'true_MC == {n}')
            # fig = plt.figure(figsize=(8, 4))
            fig = plt.figure(figsize=(4, 3))
            # fig = plt.figure(figsize=(9, 4))
            # if legend:
            #     fig = plt.figure(figsize=(9, 3))
            # else:
            #     fig = plt.figure(figsize=(3, 3))
            plot.grid_search_result(selection, x_key='n_pixelsN',
                                    y_key=y_key,
                                    z_keys=[
                                        'true_MC',
                                        'convergence_threshold',  # required for parsing
                                        # 'thread_size',
                                    ],
                                    err_key=f'{metric} std',
                                    translate=translate,
                                    # alpha_map=('convergence_threshold',
                                    #            util.make_map(convergence_thresholds)),
                                    # color_group_size=len(
                                    #     convergence_thresholds),
                                    ylog=0,
                                    y_unit='s' if metric == 'Runtime' else None,
                                    bar=1,
                                    interpolate=0,
                                    legend=legend,
                                    # loc='lower right',
                                    loc='upper left',
                                    fig=fig,
                                    # interpolate=0 if bar else 'regression',
                                    err_alpha=0., v=v)
            if n:
                plt.title('True MC')
            else:
                plt.title('Batched MC')
            # if legend:
            #     plt.legend(loc='upper right', bbox_to_anchor=(1.75, 1.1))
            # plt.ylim((0, None))
            maxi = selection.loc[:, y_key].max()
            plt.ylim(0, np.ceil(maxi * 1.05))
            plt.tight_layout()
            # plot.save(f'runtime-{bar}')
            # plot.save_fig(f'{fn}-{metric}-N-{i}', transparent=False)
            plot.save_fig(f'{fn}-{metric}-{i}', transparent=False)
    plt.close()


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
        print("TODO mu:", r'$\mathbf{\mu}=\left(_0^0\right)$')
        plt.title(f'Gaussian {mu}, $\\sigma^2={var:0.2e}$')
        plot.markup(ax, unit='m', **markup)
        # plt.suptitle('Irradiance')
        plt.tight_layout()
        plot.save_fig(f'{fn}-{d}d')


def pred_gaussian(fn: str):
    # n_sqrt = 512
    n_sqrt = 256
    build_param_values = {}
    run_param_values = {
        'n_objects': 1, 'n_pixels': n_sqrt**2, 'n_sample_points': n_sqrt**2,
        'X': 1, 'Z': 1,
        'obj_z_offset_min': [0.2, 0.3, 0.4, 0.5],
        'projector_width': np.linspace(1920 * 1e-6, 1920 * 1e-5, 5),
        'projection_width': 0.000102,
        'aspect_ratio_projection': n_sqrt**2,
        # 'randomize_pixels': True,
    }
    build_params = util.param_table(build_param_values)
    run_params = util.param_table(run_param_values)
    print('build_params', build_param_values)
    print('run_params', run_param_values)

    results, results_fn = util.get_results(build, run, build_params, run_params, fn,
                                           n_trials=1, tmp_dir=DIR,
                                           copy_data=True, v=v,
                                           find_peak_width=True)

    print('load', DIR, f'{results_fn}.zip', 'out')
    # params, data = util.parse_file(DIR, f'{results_fn}.zip', 'out')

    # widths = results.loc[:, 'peak width mean']
    print(run_params.loc[:, ['obj_z_offset_min', 'projector_width', ]])
    print(results.loc[:, [
        # 'obj_z_offset_min',
        # 'projector_width',
        'peak width mean']])

    # corr = pd.concat([run_params, results], axis=1).loc[:, [
    #     'obj_z_offset_min', 'projector_width', 'peak width mean']].corr()
    # print(corr)
    # plt.matshow(corr, vmin=-1, vmax=1, cmap='coolwarm')
    # plt.colorbar()
    # plt.show()

    results = pd.concat([run_params, results], axis=1)
    # .loc[:, [ 'obj_z_offset_min', 'projector_width', 'peak width mean']]
    # n_rows = run_params.loc[]
    # print(results.keys())
    max = results.loc[:, 'peak width mean'].max() * 1.05
    keys = ['obj_z_offset_min', 'projector_width']

    X = results.loc[:, keys].values
    y = results.loc[:, 'peak width mean'].values
    # y = y / y.max()
    print('shape')
    print(X.shape)
    print(y.shape)

    def poly(X, a, b, c, d, e, f, g):
        assert len(X.shape) == 2
        assert X.shape[1] == 2
        y = np.empty(X.shape[0])
        for i, x in enumerate(X):
            p = np.sum(np.array([a, b]) + np.array([c, d]) * x +
                       np.array([e, f]) * x ** 2)
            y[i] = p + g * np.prod(x)
        return y

    args, _ = scipy.optimize.curve_fit(poly, X, y, p0=(0, 0, 0, 0, 0, 0, 0))
    print(args)
    Xs = np.array([np.linspace(results.loc[:, k].min(),
                               results.loc[:, k].max()) for k in keys]).T

    a, b, c, d, e, f, g = args
    print(keys[0], f'\t{a:0.2f} + {c:0.2f} x + {e:0.2f} x^2 + {g:0.2f} xy')
    print(keys[1], f'\t{b:0.2f} + {d:0.2f} x + {f:0.2f} x^2 + {g:0.2f} xy')
    plt.subplot(121)
    x0 = Xs[:, 0]
    x1 = np.zeros(x0.size) + 1920 * 1e-6
    plt.plot(x0, poly(np.array([x0, x1]).T, *args))
    plt.plot(x0, poly(np.array([x0, x1]).T, *(np.round(args, 4))))
    plt.subplot(122)
    x0 = np.zeros(x0.size) + 0.3
    x1 = Xs[:, 1]
    plt.plot(x1, poly(np.array([x0, x1]).T, *args))
    plt.plot(x1, poly(np.array([x0, x1]).T, *(np.round(args, 4))))
    # plt.show()  # TODO
    plot.save_fig(f'{fn}-model', transparent=False)

    fig = plt.figure(figsize=(9, 4))
    for i, x_key in enumerate(keys):
        ax = plt.subplot(1, 2, i + 1)
        plot.grid_search_result(results,
                                x_key=x_key,
                                y_key='peak width mean',
                                z_keys=[k for k in keys if k != x_key],
                                y_unit='m', bar=False,
                                ylim=[0, max],
                                translate=translate,
                                fig=fig)

        formatter = tck.EngFormatter(places=1, sep=u"\N{THIN SPACE}", unit='m')
        ax.xaxis.set_major_formatter(formatter)
        plt.xticks(rotation=30)
    plt.tight_layout()
    plot.save_fig(f'{fn}', transparent=False)
    # plt.show()
    # plt.plot(results.loc[:, 'obj_z_offset_min'], results.loc[:, 'width'])
    # plt.plot(results.loc[:, 'projector_width'], results.loc[:, 'width'])
    # print(widths)
    # for i, row in results.iterrows():
    # print(row.loc[['peak width mean']])
    # plt.plot(data['w'][i][:, 0], data['z'][i][:, 0] ** 2)
    # for j in i_bounds:
    #     plt.axvline(w[j], ls='--', alpha=0.5)
    # plt.ylim(0, None)
    # plot.distribution1d([data['w'][i]], [data['z'][i][:, 0] ** 2],
    #                     # title,
    #                     xlog=False, ylog=False,
    #                     # figshape=(2, 2),
    #                     # labels=[f"Distance: ${p['z_offset']:.3f}$ m"
    #                     #         for p in params['z']]
    #                     )
    # plot.save_fig(f'{fn}-amp-{xlog}{ylog}', transparent=False)
    # plt.show()


if __name__ == '__main__':
    v = 1
    n_trials = 5
    prefix = 'exp'

    # print('exp matlab baseline')
    # performance_matlab(prefix + '-matlab', v=0)
    #
    # print('exp performance (3algs)')
    # performance(n_trials, prefix + '-perf0', bar=1, v=0)
    #
    # print('exp performance2e - N, threads')
    # performance2(n_trials, prefix + '-perf2e',
    #              # n_objects=np.array([1, 2, 3, 4]) * 1e6,
    #              n_objects=10**np.array([4, 5, 6, 7]),
    #              n_pixels=512 ** 2,
    #              thread_size=[(128, 128), (512, 256), (1024, 512)],
    #              bar=1
    #              )

    # print('exp performance3b - N, blocks')
    # performance3(n_trials, prefix + '-perf3b',
    #              # n_objects=np.array([1, 2]) * 1e6,
    #              n_objects=10 ** np.array([6]),
    #              # n_objects=10 ** np.array([4, 5]),
    #              n_pixels=512 ** 2,
    #              thread_size=[(512, 256)],
    #              blockDim_x=16,  # little influence
    #              blockDim_y=[8, 16, 32],
    #              gridDim_x=16,
    #              gridDim_y=[8, 16, 32]
    #              )

    # # (new)
    # # 2j, alg123, lin N, tx
    # print('performance1 2j')
    # performance1(n_trials, prefix + '-perf2j',
    #              # n_objects=np.arange(1, 6) * 256**2,
    #              # n_objects=np.arange(1, 15, 2) * 256**2,
    #              # n_objects=np.arange(1, 41, 4) * 256**2,
    #              # n_objects_options=[np.arange(1, 41, 4) * 256**2,
    #              #                    np.arange(41, 61, 4) * 256**2
    #              #                    ],
    #              n_objects_options=[np.arange(1, 41, 4) * 256**2,
    #                                 np.arange(41, 81, 4) * 256**2
    #                                 ],
    #              algorithm=[1, 2, 3],
    #              n_pixels=256**2,  # ~ 50k, to allow for many ybatches
    #              # thread_size=[(16, 16), (64, 16), (128, 16), (1024, 16)],
    #              thread_size=[(64, 8)],
    #              gridDim_x=8, blockDim_x=8,
    #              gridDim_y=8, blockDim_y=8,
    #              )

    # # n1 = 64
    # # n1 = 256
    # # performance1b(n_trials, prefix + '-perf2k-64',
    # #               nm_options=[([256**2, 512**2, 1024**2],
    # #                            np.arange(1, 41, 4) * n1**2),
    # #                           ([256**2, 512**2], np.arange(41, 61, 4) * n1**2)
    # #                           ],
    # #               thread_size=[(256, 16)],
    # #               gridDim_x=8, blockDim_x=8,
    # #               gridDim_y=8, blockDim_y=8,
    # #               )
    # n1 = 256
    # # regression fit: for increasing N: 21.39, 25.91, 39.7
    # # regression fit: with lb = 1.6mln, for increasing N: 28.44, 33.00, 47.28
    # print('performance1b 2k')
    # performance1b(n_trials, prefix + '-perf2k',
    #               nm_options=[([256**2, 512**2, 1024**2],
    #                            np.arange(1, 41, 4) * n1**2),
    #                           ([256**2, 512**2], np.arange(41, 61, 4) * n1**2)
    #                           ],
    #               thread_size=[(256, 16)],
    #               z_keys=['n_objects'],
    #               gridDim_x=8, blockDim_x=8,
    #               gridDim_y=8, blockDim_y=8,
    #               # scale_x=1,
    #               # interpolate=0,
    #               interpolate_lb=1.5e6,
    #               # interpolate_lb=1.6e6,
    #               figsize=(9, 4),
    #               )

    print('performance1b 5s streams')
    performance1b(n_trials, prefix + '-perf5s',
                  # algorithm=[1, 2, 3],
                  nm_options=[
                      ([256**2],
                       [1024 * 16]),
                      ([256**2],
                       [512**2, 1024 * 1024]),
                  ],
                  thread_size=[(512, 64)],
                  # thread_size=[(256, 16)],
                  gridDim_x=8, blockDim_x=8,
                  gridDim_y=8, blockDim_y=16,
                  # n_streams=[1, 2, 4, 5, 8, 16, 32],
                  n_streams=[1, 2, 4, 5, 8, 10, 16, 64],
                  # x_key='n_objects',
                  x_key='n_pixels',
                  z_keys=['n_streams'],
                  # alpha_key='n_streams',
                  figsize=(7, 2.5),
                  interpolate=0,
                  move_legend=1,
                  )

    # note true mc + cond mc require square N
    # mc1 show improvement for worst case - no convergence
    # zero threshold, flops, geometric N, thread_sizes, incl full estimator
    # print('exp performance MC3')
    # performance_MC3(n_trials, prefix + '-perf-mc3-thread_size', v=0)

    # mc2 show thresholds, show faster convergence
    # nonzero thresholds, no flops, not lin N because true mc requires square
    # print('exp performance MC4')
    # performance_MC4(n_trials, prefix + '-perf-mc4-', v=0)

    # cond MC, thresholds, large N
    # print('exp performance cMC')
    # performance_cMC(n_trials, prefix + '-perf-mc', v=0)
    # performance_cMC(n_trials, prefix + '-perf-mc5', n_streams=5, v=0)
    # performance_cMC(n_trials, prefix + '-perf-mc5b', n_streams=5,
    #                 # convergence_threshold=[1e-5, 1e-4, 1e-3, 1e-2],
    #                 bin_sizes=np.arange(4, 24 + 1, 2) ** 2,
    #                 v=0)
    # performance_cMC(1, prefix + '-perf-mc-s1',
    #                 # n_streams=[1, 2, 4],
    #                 # convergence_threshold=[1e-3, 1e-2],
    #                 # n_streams=[1, 2, 4, 8],
    #                 n_streams=[1,  4],
    #                 # convergence_threshold=[1e-4],
    #                 convergence_threshold=[1e-2],
    #                 # bin_sizes=np.arange(4, 22 + 1, 2) ** 2,
    #                 # bin_sizes=np.arange(4, 16 + 1, 2) ** 2,
    #                 bin_sizes=np.arange(2, 8 + 1, 1) ** 2,
    #                 z_keys=['convergence_threshold', 'n_streams'],
    #                 # color_group_size=2,
    #                 # xlim=(0, 0.3),
    #                 # ylim=(0, 8.5),
    #                 v=0)

    # print('exp distribution object')
    # n = 128 ** 2
    # projection_width = n * 7e-6 * 0.1
    # distribution_vary_object(prefix + '-dist-obj', n, projection_width,  v)
    # print('exp distribution projector')
    # distribution_vary_projector(prefix + '-dist-proj', n, v)
    # print('fit gaussian')
    # fit_gaussian(prefix + '-gaus')
    # print('predict gaussian')
    # pred_gaussian(prefix + '-gaus-pred')
