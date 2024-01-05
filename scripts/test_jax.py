#!/usr/bin/env python3

import jax
import jax.numpy as jnp

x = jnp.arange(10)
print(f'x = {x}')
print(f'type(x) = {type(x)}')

def sum_of_squares(x):
   return jnp.sum(x**2)

sum_dx = jax.grad(sum_of_squares)

x = jnp.asarray([1.0, 2.0, 3.0, 4.0, 2.0])
y = jnp.asarray([1.1, 2.1, 3.1, 4.1, 1.9])
print(f'sum_sq(x) = {sum_of_squares(x)}')
print(f'sum_dx(x) = {sum_dx(x)}')
print(f'sum_sq(y) = {sum_of_squares(y)}')
print(f'sum_dx(y) = {sum_dx(y)}')

def sum_square_error(x, y):
   return jnp.sum((x - y)**2)

sum_sq_error_dx = jax.grad(sum_square_error)

print(f'sum sq error (x, y) = {sum_square_error(x, y)}')
print(f'sum sq error dx     = {sum_sq_error_dx(x, y)}')
