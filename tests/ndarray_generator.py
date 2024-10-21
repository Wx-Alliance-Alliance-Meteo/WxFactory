from numpy import ndarray
import random
from common.device import Device

def generate_vectors(size: int, random: random.Random, min: float, max: float, devices: list[Device]) -> list[ndarray]:
    arrs: list[ndarray] = [device.xp.empty(size, dtype=float) for device in devices]

    for it in range(size):
        nb: float = random.uniform(min, max)

        for arr_it in arrs:
            arr_it[it] = nb

    return arrs

def generate_matrixes(size: tuple[int, int], random: random.Random, min: float, max: float, devices: list[Device]) -> list[ndarray]:
    arrs: list[ndarray] = [device.xp.empty(size, dtype=float) for device in devices]

    for it1 in range(size[0]):
        for it2 in range(size[1]):
            nb: float = random.uniform(min, max)

            for arr_it in arrs:
                arr_it[it1, it2] = nb
    
    return arrs
