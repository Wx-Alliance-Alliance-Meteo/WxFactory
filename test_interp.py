#! /usr/bin/env python3

import numpy
from interpolation import interpolator

from time import time

def test2D(method):
    test_array = numpy.arange(3*800*800, dtype=float).reshape(3, 800, 800)
    # print(f'test_array: \n{test_array}')
    interp = interpolator('dg', 4, 'dg', 3, 'lagrange', method)
    t0 = time()
    result_array = interp(test_array)
    t1 = time()
    # print(f'result_array: \n{result_array}')

    return t1 - t0

def test3D():
    test_array = numpy.arange(5*400*400*400, dtype=float).reshape(5, 400, 400, 400)

    # print(f'test_array: \n{test_array}')

    interp = interpolator('fv', 4, 'fv', 2, 'trilinear', ndim=3)
    result_array = interp(test_array)

    # print(f'result_array: \n{result_array}')

def main():
    t_old = test2D(0)
    t_new = 0.0 # test2D(1)
    t_alt = test2D(2)

    print(f'2D: old time {t_old:6.3f}, new time {t_new:6.3f}, alt time {t_alt:6.3f}')

    test3D()


if __name__ == '__main__':
    main()
