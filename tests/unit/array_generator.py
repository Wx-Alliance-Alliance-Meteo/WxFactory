import random

import torch

from wx_factory.context import Context


def generate_vectors(
    size: int, random: random.Random, min: float, max: float, contexts: list[Context]
) -> list[torch.Tensor]:
    """
    Generate a list of vectors

    :param size: Length of the vectors
    :param random: Randomizer to use
    :param min: Minimum of the vectors, may not be in the results
    :param max: Maximum of the vectors, may not be in the results
    :param contexts: List of contexts on which to create a vector

    :return: List of vectors. Each vector is mapped to its corresponding device in `contexts`.
             Each vector contains the same data
    """

    arrs: list[torch.Tensor] = [torch.empty(size, dtype=float, device=ctx.torch_device) for ctx in contexts]

    for it in range(size):
        nb: float = random.uniform(min, max)

        for arr_it in arrs:
            arr_it[it] = nb

    return arrs


def generate_matrices(
    size: tuple[int, int], random: random.Random, min: float, max: float, contexts: list[Context]
) -> list[torch.Tensor]:
    """
    Generate a list of matrices

    :param size: Size of the matrices
    :param random: Randomizer to use
    :param min: Minimum of the matrices, may not be in the results
    :param max: Maximum of the matrices, may not be in the results
    :param contexts: List of contexts on which to create a matrix

    :return: List of matrices. Each matrix is mapped to its corresponding device in `contexts`. Each matrix contains the same data
    """

    arrs: list[torch.Tensor] = [torch.empty(size, dtype=float, device=ctx.torch_device) for ctx in contexts]

    for it1 in range(size[0]):
        for it2 in range(size[1]):
            nb: float = random.uniform(min, max)

            for arr_it in arrs:
                arr_it[it1, it2] = nb

    return arrs
