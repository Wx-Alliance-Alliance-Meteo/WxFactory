import unittest

import torch

from wx_factory.common.tensor_math import antisymmetric_velocity_contraction, spatial_matrix_vector


class SpatialContractionTestCase(unittest.TestCase):
    def test_metric_vector_with_broadcasting(self):
        generator = torch.Generator().manual_seed(811)
        for dtype in (torch.float32, torch.float64):
            matrix = torch.randn((3, 3, 2, 1, 5), generator=generator, dtype=dtype)
            vector = torch.randn((3, 1, 4, 5), generator=generator, dtype=dtype)
            expected = torch.einsum("ij...,j...->i...", matrix, vector)
            actual = spatial_matrix_vector(matrix, vector)
            tolerance = 8 * torch.finfo(dtype).eps
            torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)

    def test_antisymmetric_contraction_and_zero_work(self):
        generator = torch.Generator().manual_seed(821)
        for dtype in (torch.float32, torch.float64):
            matrix = torch.randn((3, 3, 2, 4, 5), generator=generator, dtype=dtype)
            antisymmetric = matrix - matrix.transpose(0, 1)
            velocity = torch.randn((3, 2, 4, 5), generator=generator, dtype=dtype)
            expected = torch.einsum("j...,ij...->i...", velocity, antisymmetric)
            actual = antisymmetric_velocity_contraction(antisymmetric, velocity)
            tolerance = 8 * torch.finfo(dtype).eps
            torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
            work = (actual * velocity).sum(dim=0)
            scale = (actual * velocity).abs().sum(dim=0)
            self.assertTrue(bool((work.abs() <= 32 * torch.finfo(dtype).eps * scale).all()))
