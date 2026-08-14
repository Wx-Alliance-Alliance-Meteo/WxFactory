import math

import torch
from torch import Tensor


def vertical_interp(
    source_field: Tensor, source_levels: Tensor, dest_levels: Tensor, interp_type: str = "linear"
) -> Tensor:
    """Vertically interpolate the given source field.

    Parameters
    ----------
    source_field : torch.Tensor
        Source field with shape (num_source_levels, ...).
    source_levels : torch.Tensor
        Source level coordinates with shape (num_source_levels, ...).
    dest_levels : torch.Tensor
        Destination level coordinates with shape (num_dest_levels, ...).
    interp_type : str
        Interpolation mode. Only "linear" and "cubic" are supported.

    Returns
    -------
    torch.Tensor
        Interpolated field with shape (num_dest_levels, ...).
    """
    if source_field.ndim != 3 or source_levels.ndim != 3 or dest_levels.ndim != 3:
        raise ValueError(
            f"Expected 3D tensors with shape (num_levels, nx, ny). Got shapes \n"  # nofmt
            f"  source field:  {source_field.shape}\n"
            f"  source levels: {source_levels.shape}\n"
            f"  dest levels:   {dest_levels.shape}"
        )
    if source_field.shape[1:] != source_levels.shape[1:] or source_field.shape[1:] != dest_levels.shape[1:]:
        raise ValueError("source_field, source_levels, and dest_levels must share the leading grid dimensions.")
    if source_field.shape[0] != source_levels.shape[0]:
        raise ValueError("source_field and source_levels must have the same number of source levels.")
    interp_type = str(interp_type).lower()
    interp_types = ["linear", "cubic"]
    if interp_type not in interp_types:
        raise ValueError(f"interp_type must be one of {interp_types}")

    grid_shape = source_field.shape[1:]
    num_source_levels = source_field.shape[0]
    num_dest_levels = dest_levels.shape[0]
    device = source_field.device

    if num_source_levels < 2:
        return source_field[0, ...].unsqueeze(0).expand(*grid_shape, num_dest_levels)

    do_linear = interp_type == "linear"

    num_iter = (
        int(math.log2(num_source_levels))
        if abs(math.log2(num_source_levels) - round(math.log2(num_source_levels))) < 1e-12
        else int(math.log2(num_source_levels)) + 1
    )

    out = torch.empty(num_dest_levels, *grid_shape, dtype=torch.float64, device=device)

    for k in range(num_dest_levels):
        # Create a pair of bottom/top layers that narrow down towards the destination level
        # It does a binary search (at every point of the target level) to find which
        # source level is the closest below the destination level.
        top = torch.full(grid_shape, num_source_levels - 1, dtype=torch.int, device=device)
        bot = torch.zeros(grid_shape, dtype=torch.int, device=device)
        for _ in range(num_iter):
            ref = (top + bot) // 2
            ref_idx = ref.unsqueeze(0)
            src_mid = torch.gather(source_levels, dim=0, index=ref_idx).squeeze(0)

            mask = dest_levels[k, ...] < src_mid
            top = torch.where(mask, ref, top)
            bot = torch.where(mask, bot, ref)

        src_lvl = bot
        dst = dest_levels[k, ...]

        # If any point on the target level is less than 2 levels away from the bottom/top boundary of the
        # source, we always do a linear interpolation
        if (bot <= 0).any() or (top >= num_source_levels - 1).any() or do_linear:
            mask_surface = dst <= source_levels[0, ...]
            mask_sky = dst >= source_levels[-1, ...]
            lvl0, lvl1 = gather_level(source_levels, src_lvl), gather_level(source_levels, src_lvl + 1)
            src0, src1 = gather_level(source_field, src_lvl), gather_level(source_field, src_lvl + 1)
            delta = lvl1 - lvl0
            lin_val = (1.0 - ((dst - lvl0) / delta)) * src0 + ((dst - lvl0) / delta) * src1

            # Clamp value wherever target level goes below source ground or above source sky
            out[k, ...] = torch.where(mask_surface, source_field[0], torch.where(mask_sky, source_field[-1], lin_val))

        else:
            srcm1, src0, src1, src2 = (
                gather_level(source_levels, src_lvl - 1),
                gather_level(source_levels, src_lvl),
                gather_level(source_levels, src_lvl + 1),
                gather_level(source_levels, src_lvl + 2),
            )

            schm1, sch0, sch1, sch2 = (
                gather_level(source_field, src_lvl - 1),
                gather_level(source_field, src_lvl),
                gather_level(source_field, src_lvl + 1),
                gather_level(source_field, src_lvl + 2),
            )

            prxd = (dst - src0) / (src1 - src0)
            prda = ((sch1 - schm1) / (src1 - srcm1)) * (src1 - src0)
            prdb = ((sch2 - sch0) / (src2 - src0)) * (src1 - src0)
            prsaf = (1.0 + 2.0 * prxd) * (1.0 - prxd) * (1.0 - prxd)
            prsbf = (3.0 - 2.0 * prxd) * prxd * prxd
            prsad = prxd * (1.0 - prxd) * (1.0 - prxd)
            prsbd = (1.0 - prxd) * prxd * prxd

            out[k, ...] = sch0 * prsaf + sch1 * prsbf + prda * prsad - prdb * prsbd

    return out


def gather_level(field: Tensor, idx: Tensor):
    """Take the values of a cube of data at the given level ids.
    The levels are assumed to be indicated by the first dimension of the given data array"""

    return torch.gather(field, dim=0, index=idx.unsqueeze(0)).squeeze(0)


if __name__ == "__main__":
    source_levels = torch.zeros(5, 2, 2, dtype=torch.float64)
    source_field = torch.zeros_like(source_levels)
    for i in range(5):
        source_levels[i] = i
        source_field[i] = i**2
    dest_levels = torch.tensor(
        [[[0.5, -1.0], [2.5, 3.5]], [[1.2, 1.8], [2.0, 2.9]], [[0.75, 1.5], [3.5, 4.1]]], dtype=torch.float64
    )
    expected_lin = torch.tensor(
        [[[0.5, 0.0], [6.5, 12.5]], [[1.60, 3.40], [4.0, 8.50]], [[0.75, 2.5], [12.5, 16.0]]], dtype=torch.float64
    )
    expected_cub = torch.tensor(
        [[[0.5, 0.0], [6.5, 12.5]], [[1.44, 3.24], [4.0, 8.41]], [[0.75, 2.5], [12.5, 16.0]]], dtype=torch.float64
    )
    # print(f"Source levels: \n{source_levels}")
    # print(f"Source field: \n{source_field}")
    # print(f"dest levels: \n{dest_levels}")

    dest_field = vertical_interp(source_field, source_levels, dest_levels, "linear")
    diff_norm = (dest_field - expected_lin).norm()
    if diff_norm > 1e-15:
        print(f"Large diff in linear interpolated field: {diff_norm:.2e}\n{dest_field - expected_lin}")
        print(f"Result linear: \n{dest_field}")
        raise ValueError

    dest_field_cubic = vertical_interp(source_field, source_levels, dest_levels, "cubic")
    diff_norm = (dest_field_cubic - expected_cub).norm()
    if diff_norm > 2e-15:
        print(f"Large diff in cubic interpolated field: {diff_norm:.2e}\n{dest_field_cubic - expected_cub}")
        print(f"Result cubic: \n{dest_field_cubic}")
        raise ValueError
