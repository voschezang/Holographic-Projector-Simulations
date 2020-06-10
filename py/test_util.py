import unittest
import numpy as np

from util import *


class Test(unittest.TestCase):
    def test_pendulum(self):
        def f(*args): return list(pendulum(*args))
        args = (3,)
        assert f(*args) == [0, 1, 2, 1, 0], f"f{(*args,)}: {f(*args)}"

    def test_pendulum_range(self):
        def f(*args): return list(pendulum_range(*args))
        args = (0, 2, 1)
        assert f(*args) == [0, 1, 0], f"f{(*args,)}: {f(*args)}"
        args = (3, 0, -1)
        assert f(*args) == [3, 2, 1, 2, 3], f"f{(*args,)}: {f(*args)}"

    def test_find_nearest_denominator(self):
        for n, x, M in [(2, 1.1, 1),
                        (2, 1.9, 2),
                        (6, 1.1, 1),
                        (6, 2.1, 2),
                        (6, 3.1, 3),
                        (6, 5.9, 6),
                        (100, 49, 50),
                        (100, 16, 20),
                        (100, 99, 100)]:
            result = find_nearest_denominator(n, x)
            assert result == M, {'n': n, 'x': x, 'M': M, 'm': result,
                                 'n % M': n % M, 'n % m': n % result}

    def test_find_nearest_denominator_fuzzy(self):
        for n, x, threshold, M in [(5, 2, 0.25, 2),
                                   (5, 2.4, 0.25, 2),
                                   (5, 2.6, 0.25, 2),
                                   (5, 2.4, 0.1, 1),
                                   (5, 3.6, 0.1, 5),
                                   (5, 2.9, 0.1, 1),
                                   (11, 2, 0.091, 2),
                                   (11, 2, 0.01, 1),
                                   (11, 6, 0.091, 5)]:
            res = find_nearest_denominator(n, x, threshold)
            assert res == M, {'n': n, 'x': x, 'M': M, 'm': res,
                              'n % M': n % M, 'n % m': n % res,
                              'threshold': threshold}

    def test_superposition(self):
        n = 1
        x = np.zeros((n, 2))
        x[:, 0] = 1  # amp
        u = np.zeros((n, 3))
        v = u.copy()
        v[0, 2] = LAMBDA
        assert(norm(u - v).sum() / LAMBDA == 1.)

        i = 0
        c = np.sum(f(x[:, 0], x[:, 1], u, v[i], direction=1))
        a, phi = from_polar(c)  # actually to polar
        # TODO validate numbers
        # TODO allow error margin
        assert a == 1538461.5384615385
        assert phi == 2.4492935982947064e-16
        # print({'a': a, 'phi': phi})
        # {'a': 1538461.5384615385, 'phi': 2.4492935982947064e-16}

        x[0, :] = from_polar(1.31 - 2.1j)
        # x[0, 1] = -2.1
        v[0, 1] = 2.22e9 * LAMBDA
        v[0, 2] = 12.1 * LAMBDA
        distance = norm(u - v).sum()
        phi_next = x[0, 1] + distance * 2 * np.pi / LAMBDA
        print('amp', x[0, 0], '/', x[0, 0] / distance / LAMBDA)
        print('distance', distance, 'phi ..', phi_next)
        print(f'distance /// {distance / LAMBDA:e}')
        c = np.sum(f(x[:, 0], x[:, 1], u, v[i], direction=1))
        a, phi = from_polar(c)  # actually to polar
        # print({'a': a, 'phi': phi})

        x[0, :] = from_polar(3.33 + 4.44j)
        c = np.sum(f(x[:, 0], x[:, 1], u, v[i], direction=1))
        a, phi = from_polar(c)  # actually to polar
        print({'a': a, 'phi': phi})
        v[0, 1] = 5.51e3 * LAMBDA
        c = np.sum(f(x[:, 0], x[:, 1], u, v[i], direction=-1))
        a, phi = from_polar(c)  # actually to polar
        print({'a': a, 'phi': phi})


if __name__ == '__main__':
    # unittest.main()
    Test().test_superposition()
