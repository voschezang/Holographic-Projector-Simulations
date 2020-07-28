import numpy as np
import pandas as pd
import subprocess
import os
import collections
import matplotlib.pyplot as plt
# local
import plot
import surf
import animate
import util
from util import DIMS

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
{util.SSH} << EOF
source ~/.profile
cd cuda
nvcc -o {EXE} main.cu {flags} {arch_flags} -D{D}
EOF
"""
    return subprocess.run(content, shell=True, check=True, capture_output=True)


def run(n_objects=1, X=1, Z=0):
    assert X >= 1
    content = f"""
{util.SSH} << EOF
source ~/.profile
cd cuda
make cleanup-output
./{EXE} -x {n_objects} -X {X} -Z {Z}
EOF
"""
    return subprocess.run(content, shell=True, check=True, capture_output=True)


if __name__ == '__main__':
    v = 1
    n_trials = 2
    build_param_values = {'kernel_size': [8, 16]}
    run_param_values = {'n_objects': [8, 32, 128, 512]}
    build_params = util.param_table(build_param_values)
    run_params = util.param_table(run_param_values)
    print('build_params', build_param_values)
    print('run_params', run_param_values)

    # use multiple x/y planes and let n_trials = 1
    results = util.get_results(build, run, build_params, run_params,
                               'results-0', n_trials=n_trials, v=v)
    # results = build_params.join(run_params).join(results)
    print('done\n---\n')
    print(results)

    print('plot')
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
                print('axhline')
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
            plot.save_fig(f'{metric}-{bar}', transparent=False)
