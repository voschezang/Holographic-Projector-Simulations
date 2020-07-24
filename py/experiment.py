import numpy as np
import pandas as pd
import subprocess
import os
import matplotlib.pyplot as plt
# local
import plot
import surf
import animate
import util
from util import DIMS

EXE = 'run_experiment'
CD = 'cd ../cuda'
SSH = 'ssh nikhef-g2'


def build(kernel_size=16):
    flags = '-l curand -l cublas -std=c++14'
    arch_flags = '-arch=compute_70 -code=sm_70'
    D = f"KERNEL_SIZE='{kernel_size}'"
    content = f"""
{SSH} << EOF
source ~/.profile
cd cuda
nvcc -o {EXE} main.cu {flags} {arch_flags} -D{D}
EOF
"""
    return subprocess.run(content, shell=True, check=True, capture_output=True)


def run(n_objects):
    content = f"""
{SSH} << EOF
source ~/.profile
cd cuda
./{EXE} -x {n_objects}
EOF
"""
    return subprocess.run(content, shell=True, check=True, capture_output=True)


if __name__ == '__main__':
    build_param_values = {'kernel_size': [16, 32]}
    run_param_values = {'n_objects': [1, 2]}
    build_params = util.param_table(build_param_values)
    run_params = util.param_table(run_param_values)
    print('build_params', build_params.to_dict())
    print('run_params', run_params.to_dict())

    results = util.get_results(build, run, build_params, run_params,
                               'results-0', n_trials=2, v=0)
    results = build_params.join(run_params).join(results)
    print('done\n---\n')
    print(results.head())
    # out = subprocess.check_output('cd .. && make build', shell=True)
    # # out = build()
    # print('done\n')
    # print(out)
