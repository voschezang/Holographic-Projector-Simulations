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


def build(kernel_size=16):
    # TODO do this on remote machine, only write final results
    # fn = 'tmp_pkl/experiment.sh'
    flags = '-l curand -l cublas -std=c++14'
    arch_flags = '-arch=compute_70 -code=sm_70'
    D = f"KERNEL_SIZE='{kernel_size}'"
#     content = f"""
# ssh nikhef-g2 << EOF
# source ~/.profile
# cd cuda
# nvcc -o {EXE} main.cu {flags} {arch_flags} -D{D}
# EOF
# """
#     with open(fn, "w") as f:
#         f.write(content)

    cmd = f"{CD} && nvcc -o {EXE} main.cu {flags} {arch_flags} -D{D}"
    return subprocess.check_output(cmd, shell=True)
    # return subprocess.check_output(f'sh {fn}')
    # return subprocess.check_output(f'sh {fn}')
    # return subprocess.check_output(f'./{fn}')
    # return subprocess.check_output(f'./{fn}', shell=True)
    # return subprocess.check_output(fn, shell=True)
    # return subprocess.check_output(content)
    # return subprocess.check_output('; '.join(content.split('\n')), shell=True)
    # return subprocess.check_output(content.split('\n'), shell=True)

# make build-run
# nvcc -o {exe} main.cu {flags} {arch_flags} -D{D}


def run(n_objects):
    # return subprocess.check_output('make run')
#     fn = 'tmp_pkl/experiment.sh'
#     content = f"""
# ssh nikhef-g2 << EOF
# source ~/.profile
# cd cuda
# ./{EXE} -x {n_objects}
# EOF
# """
#     with open(fn, "w") as f:
#         f.write(content)

    cmd = f"{CD} && ./{EXE} -x {n_objects}"
    # return subprocess.check_output(f'sh {fn}')
    return subprocess.check_output(cmd, shell=True)


if __name__ == '__main__':
    build_param_values = {'kernel_size': [16, 32]}
    run_param_values = {'n_objects': [1, 2]}
    build_params = util.param_table(build_param_values)
    run_params = util.param_table(run_param_values)
    print(build_params)
    print(run_params)

    results = util.get_results(build, run, build_params, run_params,
                               'results-0', n_trials=2, v=0)
    results = build_params.join(run_params).join(results)
    print('done\n---\n')
    print(results.head())
    # out = subprocess.check_output('cd .. && make build', shell=True)
    # # out = build()
    # print('done\n')
    # print(out)
