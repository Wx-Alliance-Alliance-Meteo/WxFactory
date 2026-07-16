import types
import unittest

from wx_factory.step_hooks.registry import (
    PHASE_GEOMETRY,
    PHASE_STATE,
    STEP_HOOK_REGISTRY,
    StepHookContext,
    register_step_hook,
    resolve_step_hooks,
)

from wx_test import WxTestCase


class _FakeGeom:
    pass


def make_ctx(**config_kwargs):
    # Defaults so the built-in providers (which read these) do not raise when resolved together.
    config_kwargs.setdefault("enable_schar_mountain", False)
    config_kwargs.setdefault("case_number", 0)
    config = types.SimpleNamespace(**config_kwargs)
    return StepHookContext(config=config, geometry=_FakeGeom())


class StepHookRegistryTestCases(WxTestCase):
    def test_builtin_hooks_registered_with_expected_phases(self):
        phases = {name: phase for name, (phase, _) in STEP_HOOK_REGISTRY.items()}
        self.assertEqual(phases.get("schar_mountain"), PHASE_GEOMETRY)
        self.assertEqual(phases.get("dcmip_t11_wind"), PHASE_STATE)
        self.assertEqual(phases.get("dcmip_t12_wind"), PHASE_STATE)

    def test_resolve_only_returns_requested_phase(self):
        # schar is a geometry-phase hook; asking for the state phase must not return it.
        ctx = make_ctx(enable_schar_mountain=True, case_number=0)
        self.assertEqual(resolve_step_hooks(ctx, PHASE_STATE), {})

    def test_schar_does_not_apply_to_non_cubesphere(self):
        ctx = make_ctx(enable_schar_mountain=True, case_number=0)
        self.assertEqual(resolve_step_hooks(ctx, PHASE_GEOMETRY), {})

    def test_dcmip_dispatches_on_case_number(self):
        ctx = StepHookContext(
            config=types.SimpleNamespace(enable_schar_mountain=False, case_number=11),
            geometry=_FakeGeom(),
            operators=None,
            metric=None,
        )
        hooks = resolve_step_hooks(ctx, PHASE_STATE)
        self.assertEqual([t.__name__ for t in hooks], ["DcmipT11WindHook"])

    def test_no_hooks_when_nothing_applies(self):
        ctx = make_ctx(enable_schar_mountain=False, case_number=5)
        self.assertEqual(resolve_step_hooks(ctx, PHASE_GEOMETRY), {})
        self.assertEqual(resolve_step_hooks(ctx, PHASE_STATE), {})

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

    def test_duplicate_registration_raises(self):
        @register_step_hook("dup_hook", phase=PHASE_STATE)
        def _first(ctx):
            return None

        try:
            with self.assertRaises(ValueError):

                @register_step_hook("dup_hook", phase=PHASE_STATE)
                def _second(ctx):
                    return None

        finally:
            del STEP_HOOK_REGISTRY["dup_hook"]


if __name__ == "__main__":
    unittest.main()
