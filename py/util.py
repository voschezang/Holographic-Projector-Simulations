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
from numba import jit
from typing import Tuple
from multiprocessing import Pool
from itertools import repeat

import file
import remote


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
FLOP = 29  # per point

PROGRESS_BAR_FILL = '█'

if sys.stdout.encoding != 'UTF-8':
    try:
        PROGRESS_BAR_FILL = PROGRESS_BAR_FILL.encode('ascii')
    except UnicodeEncodeError:
        PROGRESS_BAR_FILL = '#'


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
def phasor_displacement(amplitude, phase, w, v, direction=1,  weighted=0):
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


def semilog(x):
    """ Safe log: np.ndarray
    x : float or np.array of floats
    """
    return np.log(np.clip(np.abs(x), 1e-15, None))


def semilog10(x):
    return np.log10(np.clip(np.abs(x), 1e-15, None))


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


def find_peak_widths(signal: np.ndarray, positions: np.ndarray, i_peaks=None):
    if i_peaks is None:
        i_peaks = [signal.argmax()]

    widths, _, (l,), (r,) = scipy.signal.peak_widths(
        signal, i_peaks, rel_height=0.5)
    i_bounds = np.array([l, r]).round().astype(int)
    bounds = positions[i_bounds]
    return abs(bounds[1] - bounds[0])


def cross_entropy(p: np.ndarray, q: np.ndarray):
    # both probability distributions p, q should have a sum of 1
    return - np.sum(p * np.log(q))
    # return - np.sum(p * np.log(q) + (1 - p) * np.log(1 - q))


def gaussian(x, mu1=0, mu2=0, var=1):
    # 2d but symmetric gaussian
    mu = np.array([mu1, mu2])
    # diagonal of cov matrix should be nonnegative
    var = np.eye(2) * np.clip(var, 1e-14, None)
    return scipy.stats.multivariate_normal.pdf(x, mu, var)


def gaussian_2d(x, mu1=0, mu2=0, s1=1, s2=0, s3=0, s4=1):
    mu = np.array([mu1, mu2])
    # diagonal of cov matrix should be nonnegative
    s1 = np.clip(s1, 0, None)
    s4 = np.clip(s1, 0, None)
    s = np.array([[s1, s2], [s3, s4]])
    # Note, matrix s must be positive semidefinite
    try:
        return scipy.stats.multivariate_normal.pdf(x, mu, s)
    except ValueError:
        # assume matrix s was not positive semidefinite
        s = np.array([[s1, 0], [0, s4]])
        return scipy.stats.multivariate_normal.pdf(x, mu, s)


def concat(items=[], lazy=True):
    if not lazy:
        return sum(items, [])
    return itertools.chain.from_iterable(items)


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
            f"Filesize too small ({size} < {min_size}) for {filename} (len: {n} x {precision})")

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


def parse_json(lines) -> dict:
    """ lines = list of strings containing json encoded data """
    params = {k: [] for k in 'xyzuvw'}
    for line in lines:
        if line:
            try:
                p = json.loads(line)
                k = p['phasor'][:1]
                assert k in 'xyz'
                params[k].append(p)
            except json.decoder.JSONDecodeError as e:
                print(e)
                print(line)

    return params


def parse_file(dir='../tmp', zipfilename='out.zip', prefix='out',
               read_data=True, verbose=True) -> Tuple[dict, dict]:
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

        for i, p in enumerate(concat(params.values())):
            if verbose:
                print(f'#{i}:', p)
            # TODO try read, if fail then clear params[i]
            k1 = p['phasor'][:1]
            k2 = p['pos'][:1]
            assert p['len'] == int(p['len'])
            p['len'] = int(p['len'])
            try:
                amp = _parse_doubles(p['phasor'] + '_amp.dat', p['len'],
                                     p['precision'], archive=z)
                phase = _parse_doubles(p['phasor'] + '_phase.dat', p['len'],
                                       p['precision'], archive=z)
                pos = _parse_doubles(p['pos'] + '.dat', p['len'] * p['dims'],
                                     p['precision'], archive=z)

                data[k1].append(np.array([amp, phase]).T)
                data[k2].append(pos.reshape(-1, DIMS))
                print('Amp range', amp.min(), amp.max())
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
        print(k, v)
        if isinstance(v, int) or isinstance(v, float):
            v = [v]
            params[k] = v
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
                build_params: pd.DataFrame = None, run_params: pd.DataFrame = None,
                filename='results', tmp_dir='tmp_pkl',
                n_trials=20, copy_data=False, v=0, **kwargs):
    """
    Compile & run a c-style program
    build_func (run_func) should accept build_params (run_params) and must
    return a `subprocess.CompletedProcess`
    """
    if copy_data:
        assert n_trials == 1, 'use multiple planes instead of multiple trials'
    if build_params is None:
        build_params = param_table({})
    if run_params is None:
        run_params = param_table({})
    rerun = file.get_arg("-r", default_value=False, flag=True)
    # Load or generate simulation results
    encoded = str.encode(str(build_params) + str(run_params) + str(n_trials),
                         'utf-8')
    hash = int(int(hashlib.sha256(encoded).hexdigest(), 16) % 1e12)
    filename += str(hash)
    zip_fn = f'{tmp_dir}/{filename}.zip'

    if not rerun \
            and os.path.isfile(f'{tmp_dir}/{filename}.pkl') \
            and (not copy_data or os.path.isfile(zip_fn)):
        with open(f'{tmp_dir}/{filename}.pkl', 'rb') as f:
            results = pickle.load(f)
    else:
        if v:
            print('Generate new results')
        results = grid_search(build_func, run_func, build_params, run_params,
                              n_trials=n_trials, verbose=v, **kwargs)
        print('\n')
        assert any([row['Runtime mean'] != 0 for _, row in results.iterrows()])
        with open(f'{tmp_dir}/{filename}.pkl', 'wb') as f:
            pickle.dump(results, f)

        if copy_data:
            remote.cp_file(target=zip_fn)

    return results, filename


def grid_search(build_func, run_func,
                build_params: pd.DataFrame, run_params: pd.DataFrame,
                n_trials=5, find_peak_width=False, verbose=1):
    results = []
    if verbose:
        print('Grid search')
    n_build_rows = build_params.shape[0]
    n_run_rows = run_params.shape[0]
    n_rows = max(1, n_build_rows) * n_run_rows
    for i, build_row in build_params.iterrows():
        success = True
        try:
            build_func(**build_row)
        except subprocess.CalledProcessError:
            success = False
            print('\n - Error for build params:', build_row.to_dict())

        for j, run_row in run_params.iterrows():
            # if verbose and i % (1e3 / verbose) == 0:
            #     print(f'\ni: {i}\t params: ', ', '.join(
            #         f'{k}: {v}' for k, v in run_row.items()))

            runtime, runtime_std, flops, amp, phase, peak_widths = \
                np.zeros((6, n_trials))
            params = build_row.combine_first(run_row).to_dict()
            for t in range(n_trials):
                if not success:
                    break
                idx = i * n_build_rows + j
                print_progress(idx, n_rows, t, n_trials,
                               suffix=str(params))
                try:
                    runtime[t], runtime_std[t], flops[t], amp[t], phase[t] = \
                        run_trial(run_func, run_row)

                    if find_peak_width:
                        dir = 'tmp_local'
                        fn = 'out.zip'
                        remote.cp_file(target=f'{dir}/{fn}')
                        params, data = parse_file(dir, fn, 'out', verbose=0)
                        assert len(data['z']) == 1
                        peak_widths[t] = find_peak_widths(
                            data['z'][0][:, 0] ** 2, data['w'][0][:, 0])

                except (subprocess.CalledProcessError, RuntimeError) as e:
                    success = False
                    print('\n - Error for run params:',
                          build_row.combine_first(run_row).to_dict())
                    print(e)

                    # TODO rm
                except subprocess.CalledProcessError as e:
                    success = False
                    print('\n - Error for run params:',
                          build_row.combine_first(run_row).to_dict())
                    print(e)
                except RuntimeError as e:
                    success = False
                    print('\n - Error for run params:',
                          build_row.combine_first(run_row).to_dict())
                    print(e)

            if not success:
                # invalidate all results in case of a single error for this parameter set
                runtime[:] = runtime_std[:] = flops[:] = amp[:] = phase[:] = 0
                if find_peak_width:
                    peak_widths[:] = 0
                if n_rows == 1:
                    raise RuntimeError('Single param-set experiment failed')

            # Runtime std is replaced by inter-std, but FLOPS is unchanged
            std = np.std(runtime) if n_trials > 1 else runtime_std[0]

            # Append valid and invalid results for consistency with input params
            params.update({'Runtime mean': np.mean(runtime),
                           'Runtime std': std,
                           # 'Runtime rel std': np.std(time) / np.mean(runtime),
                           'FLOPS mean': np.mean(flops),
                           'FLOPS std': np.std(flops),
                           # 'FLOPS rel std': np.std(flops) / np.mean(flops)
                           'amp std': np.std(amp),
                           'phase std': np.std(phase),
                           'amp mean': np.mean(amp),
                           'peak width mean': np.mean(peak_widths),
                           'peak width std': np.std(peak_widths),
                           })
            # std of amp/phase for all trials should be zero
            if params['amp std'] > 1e-3 or params['phase std'] > 1e-3:
                print(f"\nWarning, amp std: {params['amp std']:e},",
                      f" phase std: {params['amp std']:e}")
            results.append(params)

    return pd.DataFrame(results)


def run_trial(run_func, run_kwargs):
    """
    Return a tuple (runtime, flops).
    An `subprocess.CalledProcessError` is raised in case of incorrect params
    """
    out: subprocess.CompletedProcess = run_func(**run_kwargs)
    if out.stderr != b'':
        print(f"\nstdout = {out.stdout.decode('utf-8')} ",
              f"\n\nstderr = {out.stderr.decode('utf-8')}\n")
        raise RuntimeError('nonempty subprocess.CompletedProcess.stderr')
    out.check_returncode()
    # TODO manually cp file, don't use mounted dir
    result = parse_json(remote.read_file())
    if 'z' in result.keys():
        if not result['z']:
            del result['z']

    if 'z' in result.keys():
        runtimes = [r['runtime']
                    for r in result['z']]
    else:
        runtimes = [r['runtime']
                    for r in concat(result.values())
                    if r['runtime'] > 0.]
    if sum(runtimes) == 0:
        print('\n\n err, sum:', sum(runtimes), 'for',
              result.keys(), result, '\n\n')
        print('\n stderr: ', out.stderr.decode('utf-8'))

    k = 'z' if 'z' in result.keys() else 'y'
    runtimes = np.array([r['runtime'] for r in result[k]])
    runtime = runtimes.mean()
    std = runtimes.std()
    flops = np.mean([r['flops'] for r in result[k]])
    amp = np.mean([r['amp_sum']
                   for r in result[k]])
    phase = np.mean([r['phase_sum']
                     for r in result[k]])
    # runtime = sum((r['runtime']
    #                for r in concat(result.values())
    #                if r['runtime'] > 0.))
    # flops = np.mean([r['flops']
    #                  for r in concat(result.values())
    #                  if r['flops'] > 0.])
    # amp = np.mean([r['amp_sum']
    #                for r in result['y']])
    # phase = np.mean([r['phase_sum']
    #                  for r in result['y']])
    return runtime, std, flops, amp, phase


def bandwidth(alg, n, m, t, n_streams, thread_sizes=(1, 1), blockDim_x=1,
              gridDim_x=1, cache_threads=0, cache_blocks=0, cache_const_mem=0):
    # Effective Bandwidth
    # n,m are global source-target sizes, N,M are local sizes (batch-level)
    # non-default args must be arrays of the same shape
    # write1, read2 are skipped for algorithm 0
    tx = np.array([tup[0] for tup in thread_sizes.values])
    ty = np.array([tup[1] for tup in thread_sizes.values])
    gridDim_y = gridDim_x
    blockDim_y = blockDim_x
    GSx = blockDim_x * gridDim_x
    GSy = blockDim_y * gridDim_y
    batch_sizes = tx * GSx, ty * GSy
    n_strides_x, n_strides_y = tx / GSx, ty / GSy

    n_batches = n / batch_sizes[0] * m / batch_sizes[1]
    if cache_const_mem:
        n_batches = n / batch_sizes[0]
        # n_batches = m / ty

    if cache_blocks and cache_threads:
        print('cc', tx, ty)
        print('cc', GSx, GSy)

    N, M = batch_sizes
    # txy = tx * ty
    # txy = blockDim_x * gridDim_x
    # L2 = 6291456 / 8  # in doubles, not bytes
    # print('tx', tx)
    # print('ty', ty)
    bx, by = blockDim_x, blockDim_y  # threads per block
    gx, gy = gridDim_x, gridDim_y  # blocks per grid
    if cache_threads:
        bx = by = 1
    if cache_blocks:
        gx = gy = 1

    # if cache_const_mem:
        # n_strides_y = 1

    # note:
    # each y-thread reads the same x-values
    # each x-thread reads the same y-values
    # if the outer loop covers the source datapoints and the inner loop the
    # target datapoints, then the target datapoints are re-read an additional
    # M/GSy times
    read1 = [(alg == 0).astype(int) * (8 * N * M) * 4 / n_batches,
             # (alg == 1).astype(int) * (3 * N * M / c1 + 5 * N) * 8,
             # (alg > 1).astype(int) * (5 * M * N / c2 + 3 * M) * 8]
             (alg == 1).astype(int) * (gy * by * N * 5 + \
                                       gx * bx * M * 3 * n_strides_x) * 8,
             (alg > 1).astype(int) * (gy * by * N * 5 * n_strides_y +
                                      gx * bx * M * 3) * 8]
    write1 = [  # (alg == 0).astype(int) * 0,
        (alg == 1).astype(int) * (2 * N * M) * 8,
        (alg == 2).astype(int) * (2 * M * N / tx) * 8,
        (alg == 3).astype(int) * (2 * M * N / tx / blockDim_x) * 8]
    # / threadSize_x / threadSize_y
    read2 = write1
    write2 = [(alg == 0).astype(int) * (2 * M) * 4 / n_batches,
              (alg == 1).astype(int) * (2 * M) * 8,
              (alg > 1).astype(int) * (2 * M) * 8]

    per_batch = np.array(read1).sum(axis=0) + np.array(write1).sum(axis=0) + \
        np.array(read2).sum(axis=0) + np.array(write2).sum(axis=0)

    # per_batch = np.array(read1).sum(axis=0)

    # Note, GPU is only copyig data 1/5 of the time
    result = n_batches * per_batch / t
    assert result.shape == alg.shape
    return result


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

    print(f'\r > {round(major_iter/n_major*100,2):<4}%{minor } |{bar}|{suffix}',
          end=end)


def make_map(values=[], alpha_values=[0.5, 1, 0.8, 0.6]):
    print(alpha_values)
    return {k: v for k, v in zip(values, alpha_values)}
    # alpha_map = {values[0]: keys[0]}
    # if len(values) > 1:
    #     alpha_map[values[1]] = keys[1]
    # if len(values) > 2:
    #     alpha_map[values[2]] = keys[2]
    # if len(values) > 3:
    #     alpha_map[values[3]] = keys[3]
    # return alpha_map


def is_square(x: float):
    return float(int(x**.5)) == x ** .5
