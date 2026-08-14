"""Tests for cubed-sphere panel component conversions."""

import unittest

import torch

from tests.unit.wx_test import WxTestCase
from wx_factory.process_topology import (
    CONVERT_CONTRAVARIANT,
    CONVERT_COVARIANT,
    EAST,
    NORTH,
    SOUTH,
    WEST,
)

DIRECTIONS = ((SOUTH, "south"), (NORTH, "north"), (WEST, "west"), (EAST, "east"))


class PanelConversionTestCase(WxTestCase):
    """Structural identities the twelve seams must satisfy."""

    def setUp(self) -> None:
        self.dtype = torch.float64
        # Use a nonzero coordinate to exercise the shear terms.
        self.coord = torch.tensor(0.7, dtype=self.dtype)
        self.one = torch.tensor(1.0, dtype=self.dtype)
        self.zero = torch.tensor(0.0, dtype=self.dtype)

    def matrix(self, convert, panel: int, direction: int) -> torch.Tensor:
        """Return a conversion matrix from its action on the basis vectors."""
        first = convert[panel][direction](self.one, self.zero, self.coord)
        second = convert[panel][direction](self.zero, self.one, self.coord)
        return torch.tensor(
            [[first[0], second[0]], [first[1], second[1]]],
            dtype=self.dtype,
        )

    def test_covariant_is_the_inverse_transpose_of_contravariant(self):
        """Check that each covariant map is the inverse transpose."""
        for panel in range(6):
            for direction, name in DIRECTIONS:
                with self.subTest(panel=panel, direction=name):
                    contravariant = self.matrix(CONVERT_CONTRAVARIANT, panel, direction)
                    covariant = self.matrix(CONVERT_COVARIANT, panel, direction)
                    expected = torch.linalg.inv(contravariant).T
                    self.assertLess(float((covariant - expected).abs().max()), 1.0e-13)

    def test_conversions_preserve_the_scalar_contraction(self):
        """Check that covariant--contravariant contractions are invariant."""
        generator = torch.Generator().manual_seed(31)
        for panel in range(6):
            for direction, name in DIRECTIONS:
                with self.subTest(panel=panel, direction=name):
                    contra = torch.rand(2, generator=generator, dtype=self.dtype)
                    cov = torch.rand(2, generator=generator, dtype=self.dtype)

                    before = float(contra @ cov)
                    new_contra = CONVERT_CONTRAVARIANT[panel][direction](contra[0], contra[1], self.coord)
                    new_cov = CONVERT_COVARIANT[panel][direction](cov[0], cov[1], self.coord)
                    after = float(new_contra[0] * new_cov[0] + new_contra[1] * new_cov[1])

                    self.assertLess(abs(after - before), 1.0e-13 * max(abs(before), 1.0))

    def test_conversions_are_invertible(self):
        """A degenerate conversion would lose a component outright."""
        for panel in range(6):
            for direction, name in DIRECTIONS:
                with self.subTest(panel=panel, direction=name):
                    matrix = self.matrix(CONVERT_CONTRAVARIANT, panel, direction)
                    self.assertGreater(abs(float(torch.linalg.det(matrix))), 0.5)

    def test_conversions_are_the_identity_at_a_seam_centre(self):
        """Check that seam-centre conversions are signed permutations."""
        centre = torch.tensor(0.0, dtype=self.dtype)
        for convert in (CONVERT_CONTRAVARIANT, CONVERT_COVARIANT):
            for panel in range(6):
                for direction, name in DIRECTIONS:
                    with self.subTest(panel=panel, direction=name):
                        first = convert[panel][direction](self.one, self.zero, centre)
                        second = convert[panel][direction](self.zero, self.one, centre)
                        matrix = torch.tensor([[first[0], second[0]], [first[1], second[1]]], dtype=self.dtype)
                        # A signed permutation has one unit entry per row and column.
                        magnitudes = matrix.abs()
                        self.assertLess(float((magnitudes.sum(dim=0) - 1.0).abs().max()), 1.0e-14)
                        self.assertLess(float((magnitudes.sum(dim=1) - 1.0).abs().max()), 1.0e-14)


if __name__ == "__main__":
    unittest.main()
