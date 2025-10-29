def make_ig4(num_elem_horizontal: int, num_solpts: int) -> int:
    if num_elem_horizontal > 0x1FFFF:
        raise ValueError(f"Num elem ({num_elem_horizontal}) is too large to be encoded! (Max {0x1FFFF})")
    if num_solpts > 127 or num_solpts < 1:
        raise ValueError(f"Num solpts ({num_solpts}) is too large to be encoded (max 127)")
    return ((num_elem_horizontal & 0x1FFFF) << 7) | num_solpts


def decode_ig4(ig4: int) -> tuple[int, int]:
    return (ig4 >> 7, ig4 & 0x3F)
