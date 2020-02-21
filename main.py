import numpy as np
import subprocess
import os

import plot
import util

out = subprocess.check_output(['make', 'build-run'])
data = {}
# print('c done')
# print(out)

# example file:
# x:0,2,3,3
# y:2,3,3,4

fn = 'tmp/out.txt'
size = os.path.getsize(fn)
print(f'Input file size: {size * 1e-3:0.5f} kB')
if size > 1e6:
    print(f'WARNING, file too large: {size*1e-6} MB')

with open(fn, 'r') as f:
    for line in f:
        k, content = line.split(':')
        try:
            data[k] = np.array([complex(x) for x in content.split(',')])
        except ValueError as e:
            print('! Exception trig\n ->', e)
            for x in content.split(','):
                print(x)
                # this should crash eventually
                print(complex(x))

        # reshape to (N, 2)
        data[k] = np.array(util.from_polar(data[k])).T.reshape((-1,2))

print(data['x'].shape)

plot.matrix_multiple(data['x'], 'x', filename='img/x')
plot.matrix_multiple(data['y'], 'y', filename='img/y')
