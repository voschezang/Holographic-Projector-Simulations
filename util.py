import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import scipy.linalg
import scipy.fftpack as spfft
# import scipy.ndimage as spimg
from scipy.spatial.transform import Rotation as R
# import cvxpy as cvx
import halton
from numba import jit
from typing import Tuple, Union


LAMBDA = 0.6328e-6  # wavelength in vacuum: 632.8 nm (HeNe laser)
LAMBDA = 1
DIMS = 3
# N = 30**(DIMS - 1)
# N = 52**(DIMS - 1)
# N = 80**(DIMS - 1)
N = 100**(DIMS - 1)
PROJECTOR_DISTANCE = -4
PROJECTOR_DISTANCE = -1e3 * LAMBDA


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


def sample_grid(N, width=1, z_offset=0, random=None, center=1, dims=None, rotate_axis=None, distribution=None):
    # spatial & temporal positions
    if dims is None:
        global DIMS
        dims = DIMS

    if N <= 3:
        w = np.zeros((N, DIMS))
        if N == 2:
            w[0, 0] = width / 2
            w[1, 0] = -width / 2
        if N == 3:
            w[1, 0] = width / 2
            w[2, 0] = -width / 2
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
        w[:, :-1] += np.random.uniform(-0.5, 0.5, (N, dims - 1)) * inter_width

    w[:, -1] = z_offset
    if rotate_axis:
        # rotate every sample vector point w.r.t. axis
        return np.matmul(rotate_axis, w)
    return w


@jit(nopython=True)
def to_polar(a: np.ndarray, phi: np.ndarray):
    return a * np.exp(phi * 1j)


# @jit()
@jit(nopython=True)
def from_polar(c: np.ndarray, distance: int = 1):
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
    amplitude = origin amplitude
    phase = origin phase

    all spatial dims (except the last) for v,w must be perpendicular to the last dim

    direction = -1 or +1 and must be specified manually
    """
    # single wave superposition component
    delta = norm(w - v)
    # assert np.all(delta > 0)
    # \hat phi = A exp(\i(omega t - xt + phi))
    next_phase = phase - direction * 2 * np.pi * delta / LAMBDA
    next_amplitude = amplitude / delta
    if weighted:
        assert DIMS == 3
        weights = np.empty(delta.shape)
        n = int(np.sqrt(w.shape[0]))
        # for i, j in np.ndindex((n, n)):
        for i in range(n):
            for j in range(n):
                # indices of nearest neighbours for each w
                points = lattice_nn(w, n, i, j)
                a = points[0]
                b = points[1]
                c = points[2]
                d = points[3]
                area = quadrilateral_area(a, b, c, d)
                # area = quadrilateral_area(*lattice_nn(w, n, i, j))
                weights[idx_(n, i, j)] = area * 0.5
                # TODO return area

        return to_polar(next_amplitude, next_phase) * weights

    return to_polar(next_amplitude, next_phase)


# @jit(nopython=True)
def sum_kernel(x, w, v, direction=1, weighted=0):
    return np.sum(f(x[:, 0], x[:, 1], w, v, direction=direction, weighted=weighted))


# @jit(nopython=True)
def map_sum_kernel(x, y, w, v, direction=1, distance=1, weighted=0):
    for m in range(y.shape[0]):
        # y[m, :] = from_polar(
        #     np.sum(f(x[:, 0], x[:, 1], w, v[m], direction=direction)),
        #     distance=distance)
        y[m, :] = from_polar(
            sum_kernel(x, w, v[m], direction=direction, weighted=weighted),
            # np.sum(f(x[:, 0], x[:, 1], w, v[m], direction=direction)),
            distance=distance)


# @jit(nopython=True)
# def acc_kernel(x, w, v, direction=1, distance=1, weighted=0):
#     y = np.empty(v.shape[0])
#     # for-loop to prevent ambiguity in shape
#     return f(x[:, 0], x[:, 1], w, v[m], direction=direction, weighted=weighted)
#     return y


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


def reshape(x):
    N = np.shape(x)[0]
    d = np.sqrt(N).astype(int)
    return x.reshape((d, d))


def split_wave_vector(x):
    if DIMS == 2:
        return x[:, 0], x[:, 1]
    return reshape(x[:, 0]), reshape(x[:, 1])


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
    # Irradiance (bestralingssterkte)
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
    x[:, 0] *= 1 / np.max(x[:, 0])


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
    """ Returns a tuple rows, cols which are indices
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
    """ Returns a tuple of x,y arrays containing sample coordinates
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
