"""A custom type that describes an angle in the [-pi/2, pi/2[ interval that
can be stored as a 24-bit integer. Technically, pi/2 and -pi/2 map to the same
value, which when decoded gives -pi/2."""

import math
import typing

_INTERVAL = math.pi / 0x1000000

TAngle24 = typing.NewType("angle24", float)


def encode(f64: float) -> int:
    """Encode the given float value into a 24-bit integer.
    Input value must be in the range [-pi/2, pi/2[, otherwise they will be shifted by a multiple of `pi` to fit
    in that range."""

    # Keep in [-pi/2, pi/2[ range
    while f64 >= (math.pi / 2):
        print(f"Adjusting from {f64} to {f64 - math.pi}")
        f64 -= math.pi
    while f64 < (-math.pi / 2):
        print(f"Adjusting from {f64} to {f64 + math.pi}")
        f64 += math.pi

    # Scale, then shift, then truncate to 24 bits
    return (round(f64 / _INTERVAL) + 0x800000) & 0xFFFFFF


def decode(u64: int) -> float:
    """Decode the given 24-bit integer into a float. The resulting float is in the range [-pi/2, pi/2[.
    If the integer is larger than 24 bits, it will be truncated."""

    # Truncate to 24 bit, then shift, then scale
    return ((u64 & 0xFFFFFF) - 0x800000) * _INTERVAL


def angle24(val: float) -> float:
    """Adjust the given value to be exactly one that corresponds"""
    return decode(encode(val))
