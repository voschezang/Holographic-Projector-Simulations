import numpy as np
import scipy
import json
import skimage
import skimage.filters.rank
from skimage.morphology import disk
from skimage import io, transform
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
# import matplotlib.image as img
# import skimage.transform
# from time import time

import util

if __name__ == '__main__':
    fn = util.get_arg('-f', '../data/tom_jerry.jpg', parse_func=str)
    print('loading ', fn)
    data = io.imread(fn, as_gray=True)
    Nx, Ny = data.shape
    max_size = 40

    # texture-like sampling
    # X, Y = np.meshgrid(np.linspace(0, 1, data.shape[0]),
    #                    np.linspace(0, 1, data.shape[1]))
    # print(X.shape, Y.shape)
    # u = np.array([X.flatten(), Y.flatten(), np.zeros(data.size)]).T
    # n_samples = 10000
    # v = np.random.random((n_samples, 2))
    # y = scipy.interpolate.griddata(u[:, :2], data.T.flatten(), v)
    # indices = y.nonzero()

    # TODO extract lines/edges, then extract points

    if any(np.array(data.shape) > max_size):
        scale = max_size / max(data.shape)
        print(f'rescaling with factor {scale}')
        data = transform.rescale(data, scale)
        Nx, Ny = data.shape

    # data = skimage.filters.rank.enhance_contrast(data, disk(5))
    print(data.shape)
    fn = f"{fn[:-4]}_bw.png"
    print(f'save to {fn}')
    io.imsave(fn, data)

    x = data.T.flatten()
    print(x[0])
    X, Y = np.meshgrid(np.linspace(0, 1, data.shape[0]),
                       np.linspace(0, 1, data.shape[1]))
    print(X.shape, Y.shape)
    u = np.array([X.flatten(), Y.flatten(), np.zeros(x.size)]).T
    print('u shape', u.shape, x.shape, X.shape)

    indices = x > 0.1
    u = u[indices]
    x = x[indices]
    if 1:
        # plt.figure(figsize=(8, 4))
        # plt.subplot(121)
        # plt.title(f'{x.size} samples')
        plt.scatter(u[:, 0], u[:, 1], c=x, cmap='gray', s=1)
        # plt.subplot(122)
        # plt.title(f'{y.size} samples')
        # plt.scatter(v[:, 0], v[:, 1], c=y, cmap='gray', s=0.5)
        # plt.tight_layout()
        plt.show()

        # fig = plt.figure()
        # ax = fig.gca(projection='3d')
        # Z = data
        # surf = ax.plot_surface(X, Y, Z.T, linewidth=0, antialiased=False)
        # plt.show()

    # TODO use batches for large arrays?
    n = x.size
    # meta = {'len': n,
    #         'amp': '../tmp/x_amp.input',
    #         'pos': '../tmp/x_pos.input'}
    # print(meta)
    # with open('../tmp/in.json', 'wb') as f:
    #     f.write(json.dumps(meta).encode())

    print('size', x.size, x.size % 128)
    if u.size > 128:
        rem = x.size % 8
        if rem:
            tail_size = 8 - rem
            x = np.append(x, np.zeros(tail_size))
            u = np.append(u, np.zeros((tail_size, 3)), axis=0)

    if u.size > 128:
        assert x.size % 8 == 0, [x.size, x.size % 8]
        assert u.size % 8 == 0, [u.size, u.size % 8]

    # for phasor with zero phase, from_polar(amp) == amp + 0j
    phasor = np.array([x, np.zeros(x.size)]).T.flatten()

    with open('../tmp/x_phasor.input', 'wb') as f:
        f.write(phasor.astype('<f8').tobytes())

    with open('../tmp/x_pos.input', 'wb') as f:
        for i in range(u.shape[0]):
            f.write(u[i].astype('<f8').tobytes())
