from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from tests.unit.wx_test import WxTestCase
from wx_factory.common import Configuration, load_default_schema, readfile
from wx_factory.output.state import load_state, save_state


class StateTestCases(WxTestCase):
    def test_save_load_works(self):
        config = Configuration(readfile("config/cold_bubble.ini"), load_default_schema())
        generator = torch.Generator().manual_seed(62)
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype), TemporaryDirectory() as directory:
                original = torch.randn((5, 2, 3, 27), generator=generator, dtype=dtype)
                path = str(Path(directory) / "state")
                save_state(original, config, path)
                restored, restored_config = load_state(path)
                self.assertEqual(restored.dtype, original.dtype)
                torch.testing.assert_close(restored, original, rtol=0, atol=0)
                for field in load_default_schema().fields:
                    self.assertEqual(hasattr(restored_config, field.name), hasattr(config, field.name))
                    if hasattr(config, field.name):
                        self.assertEqual(getattr(restored_config, field.name), getattr(config, field.name))
