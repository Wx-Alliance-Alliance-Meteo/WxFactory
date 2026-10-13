import unittest

import torch
from mpi4py import MPI

from tests.integration.test_integration_state import component_errors


class IntegrationComparisonTestCase(unittest.TestCase):
    def test_component_failure_cannot_be_hidden_by_other_equations(self):
        reference = torch.ones((5, 4), dtype=torch.float64)
        actual = reference.clone()
        actual[2] *= 1.1
        errors = component_errors(reference, actual, MPI.COMM_SELF)
        self.assertAlmostEqual(max(errors), 0.1)
        self.assertEqual(errors[:2] + errors[3:], [0.0] * 4)

        reference.zero_()
        actual.zero_()
        self.assertEqual(component_errors(reference, actual, MPI.COMM_SELF), [0.0] * 5)
        actual[2, 0] = 0.1
        self.assertAlmostEqual(component_errors(reference, actual, MPI.COMM_SELF)[2], 0.1)
        actual[2, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            component_errors(reference, actual, MPI.COMM_SELF)
