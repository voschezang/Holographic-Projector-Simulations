import numpy as np
import subprocess
import zipfile
import os
import matplotlib.pyplot as plt
# local
import plot
import util
from util import DIMS


def run():
    return subprocess.check_output(['make', 'build-run'])


def parse_line(k: str, content: str):
    if k in 'uvw':
        try:
            data[k] = np.array(
                [float(x) for x in content.split(',')]).reshape(-1, DIMS)
        except ValueError as e:
            print('! Exception trig\n ->', e)
            for v in content.split(','):
                print(v)
                print(float(v))

    elif k in 'xyz':
        try:
            data[k] = np.array(
                [complex(x) for x in content.split(',')])
            # reshape to (N, 2)
            # x = np.array(util.from_polar(data[k]))
            data[k] = np.array(util.from_polar(
                data[k])).T.reshape((-1, 2))

        except ValueError as e:
            print('! Exception trig\n ->', e)
            for x in content.split(','):
                print(x)
                # this should crash eventually
                print(complex(x))


if __name__ == '__main__':
    data = {}
    if util.get_flag("-r") or util.get_flag("--rerun"):
        out = run()

    # example file:
    """
x:0,2,3,3
y:2,3,3,4
    """
    fn = 'tmp/out.zip'
    size = os.path.getsize(fn)
    print(f'Input file size: {size * 1e-3:0.5f} kB')
    if size > 1e6:
        print(f'WARNING, file too large: {size*1e-6:0.4f} MB')

    with zipfile.ZipFile(fn) as z:
        # with open(fn, 'rb') as f:
        with z.open('tmp/out.txt', 'r') as f:
            for line in f:
                k, content = line.decode().split(':')
                parse_line(k, content)

    log = util.get_flag("-log")
    if log:
        print('log abs y')

    if util.get_flag("-scatter"):
        plot.matrix_multiple(data['x'], 'x', filename='x')
        # plot.scatter_multiple(data['x'], data['u'], 'x', filename='x')
        plot.scatter_multiple(data['y'], data['v'], 'y', filename='y', s=1)
        k = 'z'
        if k in data.keys():
            plot.scatter_multiple(data['z'], data['w'], k, filename=k)
    else:
        plot.matrix_multiple(data['x'], 'x', filename='x')
        plot.matrix_multiple(data['y'], 'y', filename='y')

        k = 'z'
        if k in data.keys():
            plot.matrix_multiple(data[k], k, filename=k)

    #  plot subset
    N = data['y'].shape[0]
    N_sqrt = np.sqrt(N).astype(int)
    # n = int(5e3)
    n = int(5e3)
    # indices = np.arange(N).reshape((N_sqrt, N_sqrt))[:n, :n].flatten()
    indices = np.random.randint(0, N, n)
    plot.scatter_multiple(data['y'][indices], data['v'][indices],
                          'y', filename='y-scatter-sub', s=1)
    plot.scatter_multiple(data['z'][indices], data['w'][indices],
                          'z', filename='z-scatter-sub', s=1)

    n = indices.size
    gridsize = round(max(25, n / 5e2))
    print(f'N^2: {n}, grid: {gridsize}')
    plot.hexbin_multiple(data['y'][indices], data['v'][indices], 'y',
                         filename=f'y-hexbin', gridsize=gridsize)
    plot.hexbin_multiple(data['z'][indices], data['w'][indices], 'z',
                         filename=f'z-hexbin', gridsize=gridsize)
