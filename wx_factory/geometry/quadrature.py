import math

import scipy.special
import sympy
import torch
from torch import Tensor


def gauss_legendre(n: int) -> tuple[list[sympy.Float], Tensor, Tensor]:
    """Computes the Gauss-Legendre quadrature points (symbolic and numerical) and weights.

    Gauss-Legendre nodes are roots of the Legendre polynomial

    Arguments:
    - `n`: Number of quadrature points
    """

    # https://en.wikipedia.org/wiki/Gaussian_quadrature#Gauss%E2%80%93Legendre_quadrature

    n_digits = 34  # equivalent to quadruple precision
    if n <= 5:
        if n == 1:
            points_sym = [sympy.sympify("0")]
            weights = [2.0]
        elif n == 2:
            points_sym = [sympy.sympify("-1 / sqrt(3)"), sympy.sympify(" 1 / sqrt(3)")]
            weights = [1.0, 1.0]
        elif n == 3:
            points_sym = [sympy.sympify("-sqrt(3 / 5)"), sympy.sympify("0"), sympy.sympify(" sqrt(3 / 5)")]
            weights = [5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0]
        elif n == 4:
            points_sym = [
                sympy.sympify("-sqrt(2*sqrt(30)/35 + 3/7)"),
                sympy.sympify("-sqrt(3/7 - 2*sqrt(30)/35)"),
                sympy.sympify(" sqrt(3/7 - 2*sqrt(30)/35)"),
                sympy.sympify(" sqrt(2*sqrt(30)/35 + 3/7)"),
            ]
            weights = [
                (18.0 - math.sqrt(30.0)) / 36.0,
                (18.0 + math.sqrt(30.0)) / 36.0,
                (18.0 + math.sqrt(30.0)) / 36.0,
                (18.0 - math.sqrt(30.0)) / 36.0,
            ]
        elif n == 5:
            points_sym = [
                sympy.sympify("-sqrt(2*sqrt(70)/63 + 5/9)"),
                sympy.sympify("-sqrt(5/9 - 2*sqrt(70)/63)"),
                sympy.sympify("0"),
                sympy.sympify(" sqrt(5/9 - 2*sqrt(70)/63)"),
                sympy.sympify(" sqrt(2*sqrt(70)/63 + 5/9)"),
            ]
            weights = [
                (322.0 - 13.0 * math.sqrt(70.0)) / 900.0,
                (322.0 + 13.0 * math.sqrt(70.0)) / 900.0,
                128.0 / 225.0,
                (322.0 + 13.0 * math.sqrt(70.0)) / 900.0,
                (322.0 - 13.0 * math.sqrt(70.0)) / 900.0,
            ]
        else:
            raise ValueError(f"Invalid n = {n}")

        points_num = torch.tensor([a.evalf(n_digits, chop=True) for a in points_sym], dtype=torch.float64)
    else:
        points_num, weights = scipy.special.roots_legendre(n)
        points_sym = [sympy.Float(n, n_digits) for n in points_num]
        points_num = torch.asarray(points_num, dtype=torch.float64)

    # Python weight lists otherwise inherit PyTorch's default dtype.
    return points_sym, points_num, torch.asarray(weights, dtype=torch.float64)
