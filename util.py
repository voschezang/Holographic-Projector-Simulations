import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import scipy.linalg
import scipy.fftpack as spfft
# import scipy.ndimage as spimg
# import cvxpy as cvx
from numba import jit
from typing import Tuple, Union


LAMBDA = 0.6328e-6  # wavelength in vacuum: 632.8 nm (HeNe laser)
LAMBDA = 1
DIMS = 3
N = 50**(DIMS - 1)
N = 80**(DIMS - 1)
# N = 100**(DIMS - 1)
PROJECTOR_DISTANCE = -4
PROJECTOR_DISTANCE = -1e3 * LAMBDA


# @jit(nopython=True)
def compute_N_sqrt():
    return N if DIMS <= 2 else int(N ** (1 / (DIMS - 1)))


def sample_grid(N, width=1, z_offset=0, random=0, center=1):
    # spatial locations of sample points
    N_sqrt = compute_N_sqrt()

    if random == 'orthogonal':
        random = 0
        rows, cols = orthogonal(N)
        rows, cols = sample_dicretized(rows, cols)
        w = np.stack([rows * N_sqrt, cols * N_sqrt, np.zeros(N)],
                     axis=1)
    else:

        # w = np.stack(np.unravel_index(np.arange(N), (N_sqrt,) * (DIMS - 1)) +
        #              (np.zeros(N),), axis=1)
        # i.e. w = (range(N), range(N), (0,..)).reshape(N,3)

        # alt
        # x = np.linspace(-width / 2, width / 2, N_sqrt)
        # xx, yy = np.meshgrid(x,x)
        # w = np.stack([xx, yy, np.zeros(xx.shape)], axis=1)
        # np.stack((w, np.zeros(N)), axis=1)

        w = np.array((*np.ndindex((N_sqrt,) * (DIMS - 1)),))
        if DIMS == 2:
            w = np.stack((w[:, 0], np.zeros(N)), axis=1)
        elif DIMS == 3:
            w = np.stack((w[:, 0], w[:, 1], np.zeros(N)), axis=1)

    for i_dim in range(DIMS - 1):
        if center:
            # scale & center, then offset first dims
            # note that the N-th index points to the cell in the outer-most corner, at (N_sqrt-1, N_sqrt-1)
            w[:, i_dim] = (w[:, i_dim] / (N_sqrt - 1) - 0.5) * width
        else:
            w[:, i_dim] *= width / (N_sqrt - 1)
    # if DIMS >= 2:
    #     # scale & center, then offset first dims
    #     # note that the N-th index points to the cell in the outer-most corner, at (N_sqrt-1, N_sqrt-1)
    #     w[:, 0] = (w[:, 0] / (N_sqrt - 1) - 0.5) * width
    #
    # if DIMS >= 3:
    #     # scale & center, then offset first dims
    #     # note that the N-th index points to the cell in the outer-most corner, at (N_sqrt-1, N_sqrt-1)
    #     w[:, 1] = (w[:, 1] / (N_sqrt - 1) - 0.5) * width

    if random:
        inter_width = width / (N_sqrt - 1)
        w[:, :-1] += np.random.uniform(-0.5, 0.5, (N, DIMS - 1)) * inter_width

    w[:, -1] = z_offset
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
def f(amplitude, phase, w, v, direction=1):
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
    return to_polar(next_amplitude, next_phase)


@jit(nopython=True)
def sum_kernel(x, w, v, direction=1):
    return np.sum(f(x[:, 0], x[:, 1], w, v, direction=direction))


@jit(nopython=True)
def map_sum_kernel(x, y, w, v, direction=1, distance=1):
    for m in range(y.shape[0]):
        # y[m, :] = from_polar(
        #     np.sum(f(x[:, 0], x[:, 1], w, v[m], direction=direction)),
        #     distance=distance)
        y[m, :] = from_polar(
            sum_kernel(x, w, v[m], direction=direction),
            # np.sum(f(x[:, 0], x[:, 1], w, v[m], direction=direction)),
            distance=distance)


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
