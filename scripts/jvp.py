#!/usr/bin/env python3

import jax
import jax.numpy as jp
import numpy as np


def sq(x):
    return x*x

p, t = jax.jvp(sq, (jp.array([1.0, 2.0, 3.0, 1.0, 2.0, 3.0]),), (jp.array([2.0, 2.0, 2.0, 1.0, 1.0, 1.0]),))

print(f'p = {p}')
print(f't = {t}')


