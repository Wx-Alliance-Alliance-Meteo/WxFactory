import types
import unittest

from wx_test import WxTestCase

from wx_factory.step_hooks.registry import (
    PHASE_GEOMETRY,
    PHASE_STATE,
    STEP_HOOK_REGISTRY,
    StepHookContext,
    register_step_hook,
    resolve_step_hooks,
)


class _FakeGeom:
    pass


def make_ctx(**config_kwargs):
    # Defaults so the built-in providers (which read these) do not raise when resolved together.
    config_kwargs.setdefault("enable_schar_mountain", False)
    config_kwargs.setdefault("case_number", 0)
    config_kwargs.setdefault("expfilter_apply", False)
    config = types.SimpleNamespace(**config_kwargs)
    return StepHookContext(config=config, geometry=_FakeGeom())


class StepHookRegistryTestCases(WxTestCase):
    def test_resolve_only_returns_requested_phase(self):
        ctx = make_ctx(enable_schar_mountain=True, case_number=0)
        self.assertEqual(resolve_step_hooks(ctx, PHASE_STATE), {})

    def test_schar_does_not_apply_to_non_cubesphere(self):
        ctx = make_ctx(enable_schar_mountain=True, case_number=0)
        self.assertEqual(resolve_step_hooks(ctx, PHASE_GEOMETRY), {})

    def test_prescribed_transport_does_not_install_a_wind_hook(self):
        ctx = StepHookContext(
            config=types.SimpleNamespace(enable_schar_mountain=False, case_number=11, expfilter_apply=False),
            geometry=_FakeGeom(),
            operators=None,
            metric=None,
        )
        hooks = resolve_step_hooks(ctx, PHASE_STATE)
        self.assertEqual(hooks, {})

    def test_provider_returning_none_is_skipped(self):
        @register_step_hook("test_never", phase=PHASE_STATE)
        def _never(ctx):
            return None

        try:
            self.assertEqual(resolve_step_hooks(make_ctx(), PHASE_STATE), {})
        finally:
            del STEP_HOOK_REGISTRY["test_never"]

    def test_unknown_phase_raises(self):
        with self.assertRaises(ValueError):
            register_step_hook("bad_phase_hook", phase="not_a_phase")


if __name__ == "__main__":
    unittest.main()
