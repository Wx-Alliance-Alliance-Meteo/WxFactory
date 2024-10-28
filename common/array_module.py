import inspect
import numpy


def module_from_name(name):
    if name == "jax":
        import jax.numpy
        return jax.numpy
    elif name == "cupy" or name == "cuda":
        import cupy
        return cupy
    elif name == "numpy":
        return numpy

    return numpy


def get_array_module(array):
    mod = inspect.getmodule(type(array))
    if mod is not None:
        first = mod.__name__.split(".")[0]
        return module_from_name(first)

    return numpy
