import numpy as np
import pandas as pd
import sys
import os
import hashlib
import pickle
# import struct
# import functools
import itertools
import subprocess
import json
import zipfile
import scipy.optimize
import scipy.linalg
import halton
from numba import jit
from typing import Tuple
from multiprocessing import Pool
from itertools import repeat

import file

# LAMBDA = 0.6328e-6  # wavelength in vacuum: 632.8 nm (HeNe laser)
LAMBDA = 0.650e-6
# LAMBDA = 1
# SCALE = LAMBDA / 0.6328e-6
SCALE = 1

DIMS = 3
# N = 30**(DIMS - 1)
# N = 52**(DIMS - 1)
# N = 80**(DIMS - 1)
N = 80**(DIMS - 1)
PROJECTOR_DISTANCE = -4
PROJECTOR_DISTANCE = -1e3 * LAMBDA


PROGRESS_BAR_FILL = '█'
if sys.stdout.encoding != 'UTF-8':
    try:
        PROGRESS_BAR_FILL = PROGRESS_BAR_FILL.encode('ascii')
    except UnicodeEncodeError:
        PROGRESS_BAR_FILL = '#'

# @jit(nopython=True)
def compute_N_sqrt(n=None):
    if n is None:
        n = N
    return n if DIMS <= 2 else int(n ** (1 / (DIMS - 1)))


def halton_grid(N):
    N_sqrt = compute_N_sqrt(N)
    w = halton.halton_sequence(0, N - 1, DIMS - 1) * (N_sqrt - 1)
    return np.stack(((*w,) + (np.zeros(N),)), axis=1)


def object_grid(N):
    # amplitude, phase
    return np.zeros(shape=(N, 2))


def sample_grid(N, width=1, z_offset=0, random=None, center=1, dims=None,
                rotate_axis=None, distribution=None, HD=False):
    # spatial & temporal positions
    if HD:
        return HD_sample_grid()

    if dims is None:
        global DIMS
        dims = DIMS

    if N <= 5:
        w = np.zeros((N, DIMS))
        if N == 2:
            w[0, 0] = width / 2
            w[1, 0] = -width / 2
        if N == 3:
            w[1, 0] = width / 2
            w[2, 0] = -width / 2
        if N == 5:
            w[1, 0] = width / 3
            w[2, 0] = -width / 3
            w[3, 0] = width / 6
            w[4, 0] = -width / 6
        return w

    if random:
        # bw compatibility
        distribution = random

    # spatial locations of sample points, distributed over a plane (parallel to z by default)
    N_sqrt = compute_N_sqrt(N)
    if isinstance(distribution, np.ndarray):
        N = distribution.shape[0]
        N_sqrt = compute_N_sqrt(N)
        w = distribution.copy()
    elif distribution == 'orthogonal':
        random = 0
        rows, cols = orthogonal(N)
        rows, cols = sample_dicretized(rows, cols)
        rows *= (N_sqrt - 1)
        cols *= (N_sqrt - 1)
        w = np.stack([rows, cols, np.zeros(N)], axis=1)
        return w

    elif distribution == 'halton':
        w = halton_grid(N)

    else:
        w = np.array((*np.ndindex((N_sqrt,) * (dims - 1)),))

        if dims == 2:
            w = np.stack((w[:, 0], np.zeros(N)), axis=1)
        elif dims == 3:
            w = np.stack((w[:, 0], w[:, 1], np.zeros(N)), axis=1)

    for i_dim in range(dims - 1):
        if center:
            # scale & center, then offset first dims
            # note that the N-th index points to the cell in the outer-most corner, at (N_sqrt-1, N_sqrt-1)
            w[:, i_dim] = (w[:, i_dim] / (N_sqrt - 1) - 0.5) * width
        else:
            w[:, i_dim] *= width / (N_sqrt - 1)

    if random or distribution == 'uniform':
        # TODO if distribution == 'uniform'
        inter_width = width / (N_sqrt - 1)
        w[:, :-1] += np.random.uniform(-0.5,
                                       0.5, (N, dims - 1)) * inter_width

    w[:, -1] = z_offset
    if rotate_axis:
        # rotate every sample vector point w.r.t. axis
        return np.matmul(rotate_axis, w)
    return w


def HD_sample_grid(width=1, z_offset=0, scale=0.1, center=1, rotate_axis=None, distribution=None):
    """ Return a vector with sample coordinates of a virtual plane in 3d space.

    params
    ------
    width : scaling w.r.t. projector
    z_offset : distance from origin in third (z) dim
    scale : scale of projector/display resolution. Strongly affects runtime
    """
    N_ = round(1080 * scale)
    M_ = round(1920 * scale)
    dx = 7 * 1e-6 * width * SCALE
    # N = N_ x M_
    w = np.array((*np.ndindex((N_, M_)),))
    w = np.stack((w[:, 0], w[:, 1], np.zeros(N_ * M_)), axis=1)
    w[:, :] *= dx
    if center:
        w[:, 0] -= (N_ - 1) * dx / 2
        w[:, 1] -= (M_ - 1) * dx / 2
    if distribution:
        w[:, :-1] += np.random.uniform(-dx / 2, dx / 2, (N_ * M_, 2))

    w[:, -1] = z_offset
    if rotate_axis:
        # rotate every sample vector point w.r.t. axis
        return np.matmul(rotate_axis, w)
    return w


@jit(nopython=True)
def to_polar(a: np.ndarray, phi: np.ndarray):
    # Convert polar coordinates (a,phi) to complex number
    # TODO rename from_polar
    return a * np.exp(phi * 1j)


@jit(nopython=True)
def from_polar(c: np.ndarray, distance: int = 1):
    # TODO rename to_polar
    # Extract polar coordinates from complex number c
    # polar \in C \to (amplitude, phase)
    # sum wave superposition components
    amplitude = np.abs(c)
    phase = np.angle(c)
    if np.sign(distance) == -1:
        return amplitude, -phase
    return amplitude, phase


@jit(nopython=True)
def norm(X):
    # norm is unsupported by numba
    # return scipy.linalg.norm(v - w, ord=2, axis=-1)
    return np.sqrt(np.sum((X)**2, axis=1))


@jit(nopython=True)
def idx_(n: int, *indices):
    # index flattened nd-matrix using original indices
    # square matrix of n x n (x n)
    # \hat i = i + j * n + k * n
    idx = 0
    for k, v in enumerate(indices):
        idx += v * n ** k
    return idx


@jit(nopython=True)
def idx(x: np.ndarray, n: int, *indices):
    return x[idx_(n, *indices)]


# @jit(nopython=True)
def f(amplitude, phase, w, v, direction=1,  weighted=0):
    """
    amplitude : origin amplitude
    phase : origin phase

    all spatial dims (except the last) for v,w must be perpendicular to the
    last/prev dim

    direction : -1 or +1, must be specified manually
    """
    # single wave superposition component
    delta = norm(w - v)
    # assert np.all(delta > 0)
    # \hat phi = A exp(\i(omega t - xt + phi))
    next_phase = phase - direction * 2 * np.pi * delta / LAMBDA
    next_amplitude = amplitude / delta
    # if weighted:
    #     assert DIMS == 3
    #     weights = np.empty(delta.shape)
    #     n = int(np.sqrt(w.shape[0]))
    #     # for i, j in np.ndindex((n, n)):
    #     for i in range(n):
    #         for j in range(n):
    #             # indices of nearest neighbours for each w
    #             points = lattice_nn(w, n, i, j)
    #             a = points[0]
    #             b = points[1]
    #             c = points[2]
    #             d = points[3]
    #             area = quadrilateral_area(a, b, c, d)
    #             # area = quadrilateral_area(*lattice_nn(w, n, i, j))
    #             weights[idx_(n, i, j)] = area * 0.5
    #             # TODO return area
    #
    #     return to_polar(next_amplitude, next_phase) * weights

    return to_polar(next_amplitude, next_phase)


# @jit(nopython=True)
def sum_kernel(x, w, v, direction=1, weighted=0, plane_wave_intensity=0):

    c = np.sum(f(x[:, 0], x[:, 1], w, v,
                 direction=direction, weighted=weighted))
    # TODO why if c > 0 ?
    # if c > 0:
    #     c += f(plane_wave_intensity, 0, 0, np.ones(3).reshape(1, -1))
    if abs(plane_wave_intensity) > 0:
        c += plane_wave_intensity + 0j
    return c


def sum_kernel_wrapper(v, args):
    x, y, w,  direction, distance, weighted, plane_wave_intensity = args
    return from_polar(
        sum_kernel(x, w, v, direction=direction, weighted=weighted,
                   plane_wave_intensity=plane_wave_intensity),
        distance=distance)


# @jit(nopython=True)
def map_sum_kernel(x, y, w, v, direction=1, distance=1, weighted=0, plane_wave_intensity=0, parallel=0):
    if parallel:
        parallel = 4 if parallel in [True, 1] else parallel
        args = repeat((x, y, w, direction, distance,
                       weighted, plane_wave_intensity))
        with Pool(parallel) as p:
            results = p.starmap(sum_kernel_wrapper, zip(v, args))
        y[:] = np.array(results)
        return

    for m in range(y.shape[0]):
        y[m, :] = from_polar(
            sum_kernel(x, w, v[m], direction=direction, weighted=weighted,
                       plane_wave_intensity=plane_wave_intensity),
            distance=distance)

# @jit(nopython=True)
# def acc_kernel(x, w, v, direction=1, distance=1, weighted=0):
#     y = np.empty(v.shape[0])
#     # for-loop to prevent ambiguity in shape
#     return f(x[:, 0], x[:, 1], w, v[m], direction=direction, weighted=weighted)
#     return y


def entropy(y, v, w):
    stats = [np.abs, np.angle]
    H = np.empty((w.shape[0], len(stats)))
    for i in range(H.shape[0]):
        a = f(y[:, 0], y[:, 1], v, w[i])
        for j, func in enumerate(stats):
            hist, bin_edges = np.histogram(func(a), density=True)
            pdf = hist * (bin_edges[1:] - bin_edges[:-1])
            H[i, j] = scipy.stats.entropy(pdf)

        # hist, bin_edges = np.histogram(np.angle(a), density=True)
        # pdf = hist * (bin_edges[1:] - bin_edges[:-1])
        # H[i, 1] = scipy.stats.entropy(pdf)
    return H


def projector_setup(x, y, z, w, v, u, plane_wave_intensity=1, parallel=0):
    # projector
    assert y.shape[0] == v.shape[0]
    map_sum_kernel(x, y, w, v, direction=-1,
                   plane_wave_intensity=plane_wave_intensity, parallel=parallel)
    # for m in range(v.shape[0]):
    #     # source object
    #     c = sum_kernel(x, w, v[m], direction=-1)
    #     # plane wave
    #     c += f(plane_wave_intensity, 0, 0, np.ones(3).reshape(1, -1))
    #     y[m, :] = from_polar(c, distance=1)

    normalize_amplitude(y)

    # projection
    map_sum_kernel(y, z, v, u)
    normalize_amplitude(z)


def projector_setup2(x, w, y, v, z=None, u=None, plane_wave_intensity=0.1, parallel=0):
    # projector
    assert y.shape[0] == v.shape[0]
    for m in range(v.shape[0]):
        # source object
        y[m, :] = sum_kernel(x, w, v[m], direction=-1,
                             plane_wave_intensity=plane_wave_intensity, parallel=parallel)

    normalize_amplitude(y)
    # projection
    if z is not None:
        map_sum_kernel(y, z, v, u, parallel=parallel)
        normalize_amplitude(z)


@jit(nopython=True)
def triangle_area(a: np.ndarray, b: np.ndarray, c: np.ndarray):
    # print('tri', a.shape, b.shape, c.shape)
    ab = a - b
    ac = a - c
    # return 0.5 * np.sqrt(np.dot(a, a) * np.dot(c, c) - np.dot(b, c) ** 2)
    return 0.5 * np.sqrt(np.dot(ab, ab) * np.dot(ac, ac) - np.dot(ab, ac) ** 2)


@jit(nopython=True)
def quadrilateral_area(a, b, c, d):
    # assume convex area
    return triangle_area(a, b, c) + triangle_area(a, c, d)


@jit(nopython=True)
def lattice_nn(w, n, i, j):
    # 4 nearest points
    # TODO allow for 8 nearest points for better acc. Note, area may not be convex
    points = np.empty((4,) + w[0].shape)
    # add virtual points in case of boundaries
    i1 = i + 1 if i < n - 1 else i - 1
    j1 = j + 1 if j < n - 1 else j - 1
    i2 = i - 1 if i > 0 else i + 1
    j2 = j - 1 if j > 0 else j + 1
    points[0] = idx(w, n, i1, j)
    points[1] = idx(w, n, i, j1)
    points[2] = idx(w, n, i2, j)
    points[3] = idx(w, n, i, j2)
    return points


def vec_to_im(x):
    return reshape(x)


def reshape(x, HD=False):
    if HD:
        for scale in [0.001, 0.01, 0.02, 0.04, 0.05, 0.1, 0.2, 0.5, 1]:
            shape = round(1080 * scale), round(1920 * scale)
            if np.prod(x.shape) == np.prod(shape):
                return x.reshape(shape)
        raise NotImplementedError

    N = x.shape[0]
    d = np.sqrt(N).astype(int)
    return x.reshape((d, d))


def split_wave_vector(x, HD=0):
    if DIMS == 2:
        return x[:, 0], x[:, 1]
    return reshape(x[:, 0], HD), reshape(x[:, 1], HD)


@jit
def Fresnel_number(a, L):
    # a = aperture width, L = distance
    return a**2 / L / LAMBDA


def near(a, L):
    # a = aperture width, L = distance
    # https://en.wikipedia.org/wiki/Fresnel_diffraction
    theta = a / L
    F = Fresnel_number(a, L)
    if F * theta**2 / 4 > 1:
        print('Warning, phase terms of third order and higher must be negligible')

    if 1 <= F <= 100:
        print('Warning, F is near 1, for F = %0.2f' % F)
    return F > 1


@jit()
def irradiance(E, normalize=True):
    """ Irradiance (bestralingssterkte)
    i.e. radian flux (power) by a surface per unit area.

    Params
    ------
    E : array of phasors (complex float)
    """
    if normalize:
        a, phi = np.abs(E), np.angle(E)
        # a = 10 * (1 + a * 1 / a.max())
        a = a * 1 / a.max()
        E = to_polar(a, phi)
        # E = 100 * (10 + E * 1 / E.max())
        # E = (E * 1 / E.max()) ** (1 / 4)
    I = np.abs(E * np.conjugate(E))
    if normalize:
        I *= 1 / I.max()
        I = I ** (1 / 4)
    return I


def PSD(E):
    # power spectral density
    return irradiance(E) / N


@jit()
def normalize_amplitude(x):
    # x : array of polar coordinates (amp, phase)
    x[:, 0] /= np.max(x[:, 0])


def standardize_amplitude(x):
    # rescale (in place) to range [0,1]
    width = x[:, 0].max() - x[:, 0].min()
    x[:, 0] = (x[:, 0] - x[:, 0].min()) / width
    return x


def standardize(x):
    # normalize x and map it to the range [0, 1]
    min = x.min()
    range = x.max() - min
    if range == 0:
        return np.clip(x, 0, 1)

    return (x - min) / range


def energy_dist(a, r, xi_0, xi_r, phi_0, phi_r, theta_0, theta_r):
    # from holography handbook p. 130
    # a(x,y), r(x,y) are amplitude variations at object and ref
    # phi_i are phase variations
    # theta_i are angles at which the wave propagates
    xi_0 = np.sin(theta_0) / LAMBDA
    xi_r = np.sin(theta_r) / LAMBDA
    return a**2 + r**2 + 2 * a * r * np.cos(2 * np.pi * (xi_0 - xi_r) * x + phi_0 - phi_r)


def K_dft(n, k, N):
    #     N = compute_N_sqrt()
    return np.exp(-2j * np.pi * n * k / N)


def K(n, k, width1=1, width2=1, offset=0, N_sqrt=100):
    #     N_sqrt = compute_N_sqrt()
    dx1 = 1 / width1
    dx2 = 1 / width2
    shape = (N_sqrt,) * (DIMS - 1)
    # TODO add v,w to input
    v = np.array(np.unravel_index([n], shape) + (0,))
    w = np.array(np.unravel_index([k], shape) + (offset,))
    v[:2] *= dx1
    w[:2] *= dx2
#     v = w = np.array([0,1])
    delta = scipy.linalg.norm(v - w, ord=2, axis=-1)
    return np.exp(-2j * np.pi * delta / LAMBDA)


@jit(nopython=True)
def orthogonal(n_samples: int):
    """ Return a tuple rows, cols which are indices
    """
    major = int(round(np.sqrt(n_samples)))
    # note that major^2 could differ from n_samples
    if n_samples != major ** 2:
        print("Warning, n samples must be square")
    n_samples = major ** 2
    # generate indices and shuffle rows, while maintaining colum order
    # I,J are matrices representing indices i,j of dims x,y
    row_index_matrix = np.arange(n_samples).reshape(major, major)
    col_index_matrix = row_index_matrix.copy()
    np.random.shuffle(row_index_matrix)
    np.random.shuffle(col_index_matrix)
    # note that the y indices (matrix J) is transposed
    rows, cols = row_index_matrix.flatten(),  col_index_matrix.T.flatten()
    return rows, cols


@jit(nopython=True)
def sample_dicretized(rows, cols, n_bins=0):
    """ Return a tuple of x,y arrays containing sample coordinates
    """
    # using a 2D array raises numba error: NotImplementedError: iterating over 2D array
    # x, y = np.random.uniform(0, 1, size=(2, n_samples))
    # TODO rm arg n_samples
    n_samples = rows.size

    if n_bins == 0:
        # default for Latin Hypercube, Orthogonal
        n_bins = n_samples

    x = np.random.uniform(0, 1, size=n_samples)
    y = np.random.uniform(0, 1, size=n_samples)
    return scale_xy(rows[:n_samples], cols[:n_samples], x, y, n_bins)


@jit(nopython=True)
def scale_xy(rows, cols, x, y, n_bins) -> Tuple[np.ndarray, np.ndarray]:
    """
    Assume width = height = 1
    x \equiv i * dx
    y \equiv j * dy
    """
    dx = 1 / n_bins
    dy = 1 / n_bins
    x_scaled = (x + rows) * dx
    y_scaled = (y + cols) * dy
    return x_scaled, y_scaled


def stratified(n_samples: int, n_subspaces: int):
    """ Basic stratified sampling
    """
    v = 0
    n_bins = round(n_subspaces ** 0.5)
    n_subspaces = n_bins ** 2
    subspace_shape = (n_bins, n_bins)
    if v:
        print('n_subspaces ', n_subspaces)

    # subspace_sample_distribution = np.ones(n_subspaces) / n_subspaces
    n_samples_per_subspace = round(n_samples / n_subspaces)
    assert n_samples_per_subspace > 0, 'not enough samples'

    if v:
        print('n_samples_per_subspace', n_samples_per_subspace)

    indices = np.repeat(np.arange(n_subspaces), n_samples_per_subspace)
    # cannot use np.random.choice, as it would result in pure random sampling
    # indices = np.random.choice(
    #     np.arange(n_subspaces), size=n_samples, p=subspace_sample_distribution)
    return indices, subspace_shape


def semilog(x):
    """ Save log: np.ndarray
    x : float or np.array of floats
    """
    return np.log(np.clip(np.abs(x), 1e-12, None))


def pendulum(n: int):
    """ Returns a back-and-forth iterator.
    E.g. `list(pendulum_range(3)) = [0,1,2,3,2,1,0]`
    """
    # iterate forwards
    for i in range(n):
        yield i
    for i in range(n - 2, -1, -1):
        yield i


def pendulum_range(*args):
    """ Returns a back-and-forth version of `range` (or vice versa).
    Accepts the same arguments: `[start], stop[, step]`
    E.g. `list(pendulum_range(3)) = [0,1,2,3,2,1,0]`
    """
    # Derive all args from (start, stop, step)
    if len(args) == 1:
        args = (0,) + args + (1,)
    elif len(args) == 2:
        args += (1,)

    # flip sign if counting backwards
    pm = +1 if args[2] < 0 else -1

    # iterate forwards
    for i in range(*args):
        yield i

    # iterate backwards
    for i in range(args[1] + pm * 2, args[0] + pm, -args[2]):
        yield i


def soft_round(matrix: np.ndarray, threshold=1e-9, decimals=9, verbose=0):
    """ Round values in `matrix` if data range is below `threshold`.
    """
    minmax = np.array([matrix.min(), matrix.max()])
    # if range == 0:
    # return np.ones_like(matrix) * matrix.min()
    # s = standardize(matrix)
    range = minmax[1] - minmax[0]
    # print('range', range, range < threshold)
    # abs_min = minmax.abs().min()
    # min = minmax.abs().min()
    # if range != 0 and range / min < threshold:
    if range < threshold:
        if verbose:
            print('range < threshold', verbose)
        # normalize and rm noise
        return matrix.round(decimals)
        # matrix = (matrix / min).round(decimals)
        # # scale back
        # return matrix * min

    return matrix


def find_nearest_denominator(n: int, x: float, threshold=0):
    """ Return min_m (x - m) under the constraint n % m == 0
    e.g. for n = 10, x = 5.1 return 5

    threshold can be used in case of primes
    e.g. a theshold of 0.01 will allow for an 10% error w.r.t n

    in case x-1 and x+1 are equally valid results, the former is chosen
    """
    if x >= n:
        return n

    assert n >= x > 0, f'for n {n}, x: {x}'
    assert int(n) == n
    maximum = threshold * n  # zero if threshold is zero
    m = round(x)
    if n % m <= maximum:
        return m

    m1 = max(1, np.floor(x).astype(int))
    m2 = np.ceil(x).astype(int)

    if m1 == m2 and maximum > 0:
        # consider both m-1 and m+1 and the resulting relative error
        # note that in case of zero threshold there is no reason to prefer m2 over m1
        m1 -= 1
        m2 += 1
        # use the closest value; m1 is compared first
        error1 = n % m1
        error2 = n % m2
        options = [(m1, error1), (m2, error2)]
        if error2 < error1:
            options.reverse()
        for m, error in options:
            if error < maximum:
                return m

    elif m1 != m2:
        # consider m1 and m2, but ordered
        options = [m1, m2]
        if n - m2 < n - m1:
            options.reverse()

        for m in options:
            if n % m <= maximum:
                return m

    while m1 >= 0 or m2 <= n:
        # bias for m1
        if m1 >= 0:
            if n % m1 <= maximum:
                return m1
            else:
                m1 -= 1

        if m2 <= n:
            if n % m2 <= maximum:
                return m2
            else:
                m2 += 1

    # alt, in case x in [m-1, m+1]
    # m1 = int(n / np.floor(n / x))
    # m2 = int(n / np.ceil(n / x))
    # if abs(x - m2) < abs(x - m1):
    #     # use the closest value; m1 is compared first
    #     m1, m2 = m2, m1
    #
    # if n % m1 == 0:
    #     return m1
    # elif n % m2 == 0:
    #     return m2
    #
    raise NotImplementedError(f'Cannot find result for args {n} / {x}')


def regular_bin_edges(minima=[], maxima=[], n_bins=[]):
    n = len(n_bins)
    assert (n == len(minima)) and (n == len(maxima))
    # return [np.linspace(minima[i], maxima[i], n_bins[i] + 1)[1:-1] for i in range(n)]
    return [np.linspace(minima[i], maxima[i], n_bins[i] + 1) for i in range(n)]


def gen_bin_edges(x, y, ratio=1., ybins=10, bin_threshold=0.1, options={},
                  verbose=0):
    """ Generate uniform bins with some aspect ratio
    """
    assert ratio != 0
    assert x.shape == y.shape
    Nx, Ny = solve_xy_is_a(x.size, ratio)
    ybins = int(find_nearest_denominator(Ny, ybins, bin_threshold))
    if Nx == Ny:
        # ratio is close to 1.0
        xbins = ybins
        assert xbins <= Nx
    else:
        # derive xbins from updated ybins to preserve "squareness" of pixels
        xbins = min(Nx, round(ybins * ratio))
        xbins = find_nearest_denominator(Nx, xbins, bin_threshold)

    if verbose:
        print('bins:',
              f'\tx: {xbins} (~{Nx / xbins:0.1f} per bin)\n',
              f'\ty: {ybins} (~{Ny / ybins:0.1f} per bin)')

    assert xbins <= Nx
    assert ybins <= Ny
    # TODO implement proper interface instead of bin_options dict
    try:
        x_center, y_center = options['x_offset'], options['y_offset']
        width = options['width']
        height = width / options['aspect_ratio']
        xmin = x_center - width / 2.
        xmax = x_center + width / 2.
        ymin = y_center - height / 2.
        ymax = y_center + height / 2.
    except KeyError:
        xmin, xmax, ymin, ymax = x.min(), x.max(), y.min(), y.max()

    # bins = (np.linspace(x.min(), x.max(), xbins + 1)[1:-1],
    #         np.linspace(y.min(), y.max(), ybins + 1)[1:-1])
    return regular_bin_edges([xmin, ymin],
                             [xmax, ymax],
                             [xbins, ybins])


def solve_xy_is_a(n: int, ratio=1.):
    """ Solve the equation x * y = n, where x,y are unkown, for a known ratio x/y.
    Can be used to find dimensions of a flattened matrix
    (with original shape (x,y)).
    """
    x = int(np.sqrt(n * ratio))
    y = n // x
    assert x * y <= n
    if ratio == 1.:
        assert x == y

    return x, y


def fileinfo(archive: zipfile.ZipFile, filename: str):
    return next(filter(lambda x: x.filename == filename, archive.infolist()))


def get_flag(name: str):
    return get_arg(name, False, True)


def get_arg(name: str, default_value=None, flag=False, parse_func=int):
    """ Parse command line args """
    try:
        index = sys.argv.index(name)
        value = sys.argv[index + 1]
    except ValueError:
        if default_value is None:
            raise NameError("Unable to find argument: {}".format(name))
        else:
            value = default_value

    except IndexError:
        if flag:
            value = True
        else:
            raise IndexError("No value found for argument: {}".format(name))

    return parse_func(value)


def parse_line(data: dict, k: str, content: str):
    """
    data : dict of {char: [np.array]}
    """
    if k in 'uvw':
        try:
            data[k].append(np.array(
                [float(x) for x in content.split(',')]).reshape(-1, DIMS)
            )
        except ValueError as e:
            print('! Exception trig\n ->', e)
            for v in content.split(','):
                print(v)
                print(float(v))

    elif k in 'xyz':
        try:
            complex_values = np.array(
                [complex(x) for x in content.split(',')])
            # reshape to (N, 2)
            # x = np.array(util.from_polar(data[k]))
            data[k].append(
                np.array(from_polar(complex_values)).T.reshape((-1, 2)))

        except ValueError as e:
            print('! Exception trig\n ->', e)
            for x in content.split(','):
                print(x)
                # this should crash eventually
                print(complex(x))


def _parse_complex(filename: str, n: int, precision=8):
    re, im = _parse_doubles(filename, n, precision).reshape((-1, 2))
    return re + im * 1j


def _parse_doubles(filename: str, n: int, precision=8, sep='',
                   archive: zipfile.ZipFile = None):
    if precision not in (4, 8, 16):
        raise NotImplementedError

    if archive is None:
        size = os.path.getsize(filename)
    else:
        size = fileinfo(archive, filename).file_size

    # TODO use sep length in bytes
    min_size = n * precision
    if size < min_size:
        raise SystemError(
            f"Filesize too small ({size} < {min_size}) for {filename}")

    if sep:
        # use text (csv) files because binary files are not platform independent
        # but it does allow for scientific notation with arbitrary precision
        # comma separator results in more data but is required for correct output
        data = np.fromfile(filename, count=n, sep=sep, dtype=f"<f{precision}")
    else:
        # interpret as packed binary data
        if archive is None:
            open_func, mode = open, 'rb'
        else:
            open_func, mode = archive.open, 'r'

        with open_func(filename, mode) as f:
            data = np.frombuffer(f.read(n * precision),
                                 dtype=f'<f{precision}', count=n)

    assert data.size == n, \
        f"{filename}\tsize is {data.size} but should've been {n}"
    return data

def parse_json(fn='out.json') -> dict:
    params = {k: [] for k in 'xyzuvw'}
    with open(fn, 'r') as f:
        for line in f:
            if line:
                p = json.loads(line)
                k = p['phasor'][:1]
                assert k in 'xyz'
                params[k].append(p)
    return params




def parse_file(dir='../tmp', zipfilename='out.zip', prefix='out', read_data=True) -> Tuple[dict, dict]:
    """ Returns two tuples params and data """
    # TODO use better datastructure
    params = {k: [] for k in 'xyzuvw'}
    data = {k: [] for k in 'xyzuvw'}
    with zipfile.ZipFile(os.path.join(dir, zipfilename)) as z:
        filename = prefix + '.json'
        with z.open(filename, 'r') as f:
            for line in f:
                if line:
                    p = json.loads(line)
                    k = p['phasor'][:1]
                    assert k in 'xyz'
                    params[k].append(p)
                    # params.append(json.loads(line))

        if not read_data:
            return params,

        for i, p in enumerate(itertools.chain.from_iterable(params.values())):
            print(i, p)
            # TODO try read, if fail then clear params[i]
            k1 = p['phasor'][:1]
            k2 = p['pos'][:1]
            try:
                amp = _parse_doubles(p['phasor'] + '_amp.dat', p['len'],
                                     p['precision'], archive=z)
                phase = _parse_doubles(p['phasor'] + '_phase.dat', p['len'],
                                       p['precision'], archive=z)
                pos = _parse_doubles(p['pos'] + '.dat', p['len'] * p['dims'],
                                     p['precision'], archive=z)

                data[k1].append(np.array([amp, phase]).T)
                data[k2].append(pos.reshape(-1, DIMS))
                for x in (amp, phase, pos):
                    assert(not np.isnan(x.sum()))
                    assert(not np.isnan(x.min()))
                    assert(not np.isnan(x.max()))

            except SystemError as e:
                print(f'Warning, missing data for keys: {k1}, {k2}')
                print(e)
                p['len'] = 0

    return params, data


def param_table(params: dict) -> pd.DataFrame:
    """ Convert set (dict) of unique param values to rows of param values.
    params = {str: list of values}
    """
    for k, v in params.items():
        assert(isinstance(k, str))
        assert(isinstance(v, np.ndarray) or isinstance(v, list))

    outer = itertools.product(*params.values())
    rows = [param_row(params.keys(), values) for values in outer]
    return pd.DataFrame(rows)


def param_row(param_keys, param_values):
    """ Returns a dict {param key: param value}
    """
    values = list(param_values)
    return {k: values[i] for i, k in enumerate(param_keys)}


def get_results(build_func, run_func,
                build_params: pd.DataFrame, run_params: pd.DataFrame,
                filename='results', tmp_dir='tmp_pkl',
                n_trials=20, result_indices=[-2, -1],
                v=0):
    load_result = file.get_arg("--load_result", default_value=False, flag=True)
    # Load or generate simulation results
    # encoded = (str(build_params) + str(run_params)).encode('utf-8')
    encoded = str.encode(str(build_params) + str(run_params), 'utf-8')
    hash = int(int(hashlib.sha256(encoded).hexdigest(), 16) % 1e6)
    filename += str(hash)
    if load_result and os.path.isfile(f'{tmp_dir}/{filename}.pkl') and 0:
        with open(f'{tmp_dir}/{filename}.pkl', 'rb') as f:
            results = pickle.load(f)

        # with open(f'{tmp_dir}/{filename}-build-params.pkl', 'rb') as f:
        #     build_params = pickle.load(f)

        # with open(f'{tmp_dir}/{filename}-run-params.pkl', 'rb') as f:
        #     run_params = pickle.load(f)

    else:
        if not os.path.isfile(f'{tmp_dir}/{filename}.pkl'):
            print('Warning, results file not found')

        # if not os.path.isfile(f'{tmp_dir}/{filename}-build-params.pkl'):
        #     print('Warning, build params file not found')

        # if not os.path.isfile(f'{tmp_dir}/{filename}-run-params.pkl'):
        #     print('Warning, run params file not found')

        print('Generate new results')
        results = grid_search(build_func, run_func, build_params, run_params,
                              n_trials=n_trials, result_indices=result_indices, verbose=v)
        if v:
            print("results:\n", build_params.join(run_params).join(results))

        # Save results
        with open(f'{tmp_dir}/{filename}.pkl', 'wb') as f:
            # TODO join with params?
            pickle.dump(results, f)

        with open(f'{tmp_dir}/{filename}-params.pkl', 'wb') as f:
            pickle.dump(build_params, f)

        with open(f'{tmp_dir}/{filename}-params.pkl', 'wb') as f:
            pickle.dump(run_params, f)

    return results


def grid_search(build_func, run_func,
              build_params: pd.DataFrame, run_params: pd.DataFrame,
              n_trials=5, result_indices=[-2, -1], verbose=1):
    results = []
    if len(result_indices) == 1:
        # compatiblity
        result_indices.append(result_indices[0])

    if verbose:
        print('Grid search')
    n_build_rows = build_params.shape[0]
    n_run_rows = run_params.shape[0]
    n_rows = n_build_rows * n_run_rows
    for i, build_row in build_params.iterrows():
        build_func(**build_row),
        for j, run_row in run_params.iterrows():
            if verbose and i % (1e3 / verbose) == 0:
                print(f'i: {i}\t params: ', ', '.join(
                    f'{k}: {v}' for k, v in run_row.items()))

            time, flops = np.empty((2, n_trials))
            params = build_row.combine_first(run_row).to_dict()

            # run n independent trials
            for t in range(n_trials):
                idx = i * n_build_rows + j
                print_progress(idx, n_rows, t, n_trials, suffix=str(params))
                try:
                    run_func(**run_row)
                    out = parse_json('../tmp/out.json')
                    values = lambda: itertools.chain.from_iterable(out.values())
                    time = [r['runtime'] for r in values() if r['runtime'] > 0.]
                    flops = [r['flops'] for r in values() if r['flops'] > 0.]

                except subprocess.CalledProcessError as e:
                    print('\n  Error for params:', build_row.combine_first(run_row).to_dict())
                    time = [0]
                    flops = [0]

                # print(out.values())
                # values = sum(raw.values(), [])

            # flops *= 1e-9
            results.append({'Runtime mean': np.mean(time),
                            'Runtime std': np.std(time),
                            # 'Runtime rel std': np.std(time) / np.mean(time),
                            'FLOPS mean': np.mean(flops),
                            'FLOPS std': np.std(flops),
                            # 'FLOPS rel std': np.std(flops) / np.mean(flops)
                            })

    print('')  # close progress bar
    return pd.DataFrame(results)


def print_progress(major_iter=0, n_major=100, minor_iter=0, n_minor=None,
                   nofill='-', bar_len=30, suffix_len=70, suffix='',
                   end='\r'):
    fill_length = round(bar_len * major_iter / n_major)
    bar = PROGRESS_BAR_FILL * fill_length + nofill * (bar_len - fill_length)
    minor = ''
    if n_minor is not None:
        minor = f' ({minor_iter:<4}/{n_minor})'

    if len(suffix) > 1:
        suffix = f' ({suffix[:suffix_len]}{".." if len(suffix) > suffix_len else ")"}'

    print(f'\r > {round(major_iter/n_major * 100 ,3):<6}%{minor } |{bar}|{suffix}',
          end=end)
