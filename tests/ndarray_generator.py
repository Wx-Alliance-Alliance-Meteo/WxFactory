from numpy import ndarray
import random
from common.device import Device


def generate_vectors(size: int, random: random.Random, min: float, max: float, devices: list[Device]) -> list[ndarray]:
    """
    Generate a list of vectors

    :param size: Length of the vectors
    :param random: Randomizer to use
    :param min: Minimum of the vectors, may not be in the results
    :param max: Maximum of the vectors, may not be in the results
    :param devices: List of device to create a vector on

    :return: List of vectors. Each vector is mapped to its corresponding device in `devices`. Each vector contains the same data
    """

    arrs: list[ndarray] = [device.xp.empty(size, dtype=float) for device in devices]

    for it in range(size):
        nb: float = random.uniform(min, max)

        for arr_it in arrs:
            arr_it[it] = nb

    return arrs


def generate_matrixes(
    size: tuple[int, int], random: random.Random, min: float, max: float, devices: list[Device]
) -> list[ndarray]:
    """
    Generate a list of matrixes

    :param size: Size of the matrixes
    :param random: Randomizer to use
    :param min: Minimum of the matrixes, may not be in the results
    :param max: Maximum of the matrixes, may not be in the results
    :param devices: List of device to create a matrix on

    :return: List of matrixes. Each matrix is mapped to its corresponding device in `devices`. Each matrix contains the same data
    """

    arrs: list[ndarray] = [device.xp.empty(size, dtype=float) for device in devices]

    for it1 in range(size[0]):
        for it2 in range(size[1]):
            nb: float = random.uniform(min, max)

            for arr_it in arrs:
                arr_it[it1, it2] = nb

    return arrs
