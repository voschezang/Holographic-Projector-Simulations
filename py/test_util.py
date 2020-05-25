import unittest

from util import *


class Test(unittest.TestCase):
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
            assert res == M,  \
                {'n': n, 'x': x, 'M': M, 'm': res,
                 'n % M': n % M, 'n % m': n % res, 'threshold': threshold}


if __name__ == '__main__':
    unittest.main()
