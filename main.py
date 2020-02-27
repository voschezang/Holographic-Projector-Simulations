import numpy as np
import subprocess
import os
import matplotlib.pyplot as plt
# local
import plot
import util


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
            try:
                data[k] = np.array([complex(x) for x in content.split(',')])
            except ValueError as e:
                print('! Exception trig\n ->', e)
                for x in content.split(','):
                    print(x)
                    # this should crash eventually
                    print(complex(x))

            # reshape to (N, 2)
            data[k] = np.array(util.from_polar(data[k])).T.reshape((-1, 2))

    print(data['x'].shape)

    log = util.get_flag("-log")
    if log:
        print('log abs y')
    plot.matrix_multiple(
        data['x'], 'x', filename='img/x', interpolation='none', log=0)
    plot.matrix_multiple(
        data['y'], 'y', filename='img/y', interpolation='nearest', log=log)
