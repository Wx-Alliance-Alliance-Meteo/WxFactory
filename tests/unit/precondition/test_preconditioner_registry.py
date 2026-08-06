import types
import unittest

from wx_test import WxTestCase

from wx_factory.precondition import (
    PRECONDITIONER_REGISTRY,
    Preconditioner,
    register_preconditioner,
    resolve_preconditioner,
)


def make_ctx(name):
    """Minimal context: resolve_preconditioner only reads config.preconditioner."""
    return types.SimpleNamespace(config=types.SimpleNamespace(preconditioner=name))


class PreconditionerRegistryTestCases(WxTestCase):
    def test_none_resolves_to_no_preconditioner(self):
        self.assertIsNone(resolve_preconditioner(make_ctx("none")))

    def test_no_builtin_preconditioners(self):
        # The broken historical preconditioners were removed; the registry starts empty.
        self.assertEqual(PRECONDITIONER_REGISTRY, {})

    def test_unknown_preconditioner_raises_helpful_error(self):
        with self.assertRaises(ValueError) as cm:
            resolve_preconditioner(make_ctx("does_not_exist"))
        self.assertIn("Unknown preconditioner", str(cm.exception))

    def test_resolve_dispatches_to_registered_factory(self):
        sentinel = object()

        @register_preconditioner("test_precond")
        def _factory(ctx):
            return sentinel

        try:
            self.assertIs(resolve_preconditioner(make_ctx("test_precond")), sentinel)
        finally:
            del PRECONDITIONER_REGISTRY["test_precond"]

    def test_duplicate_registration_raises(self):
        @register_preconditioner("dup_precond")
        def _first(ctx):
            return None

        try:
            with self.assertRaises(ValueError):

                @register_preconditioner("dup_precond")
                def _second(ctx):
                    return None

        finally:
            del PRECONDITIONER_REGISTRY["dup_precond"]

    def test_base_prepare_is_a_noop(self):
        # A preconditioner that does not override prepare must still be callable by integrators.
        self.assertTrue(hasattr(Preconditioner, "prepare"))


if __name__ == "__main__":
    unittest.main()
