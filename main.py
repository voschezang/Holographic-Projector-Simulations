import numpy as np
import subprocess
import os
import matplotlib.pyplot as plt
# local
import plot
import util
from util import DIMS


def run():
    return subprocess.check_output(['make', 'build-run'])


if __name__ == '__main__':
    data = {}
    if util.get_flag("-r") or util.get_flag("--rerun"):
        out = run()

    # example file:
    """
x:0,2,3,3
y:2,3,3,4
    """
    fn = 'tmp/out.txt'
    size = os.path.getsize(fn)
    print(f'Input file size: {size * 1e-3:0.5f} kB')
    if size > 1e6:
        print(f'WARNING, file too large: {size*1e-6:0.4f} MB')

    with open(fn, 'rb') as f:
        for line in f:
            k, content = line.decode().split(':')
            if k in 'uvw':
                data[k] = np.array(
                    [float(x) for x in content.split(',')]).reshape(-1, DIMS)

            elif k in 'xyz':
                try:
                    data[k] = np.array(
                        [complex(x) for x in content.split(',')])
                    # reshape to (N, 2)
                    xx = np.array(util.from_polar(data[k]))
                    data[k] = np.array(util.from_polar(
                        data[k])).T.reshape((-1, 2))

                except ValueError as e:
                    print('! Exception trig\n ->', e)
                    for x in content.split(','):
                        print(x)
                        # this should crash eventually
                        print(complex(x))

    log = util.get_flag("-log")
    if log:
        print('log abs y')

    if util.get_flag("-scatter"):
        plot.scatter_multiple(data['x'], data['u'], 'x', filename='x', log=log)
        plot.scatter_multiple(data['y'], data['u'], 'y', filename='y', log=log,
                              s=1)
    else:
        plot.matrix_multiple(data['x'], 'x', filename='x',
                             interpolation='none', log=0)
        plot.matrix_multiple(data['y'], 'y', filename='y',
                             interpolation='nearest', log=log)
