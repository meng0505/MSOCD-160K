"""Small CPU checks for the released fusion modules and model configurations."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from ultralytics.nn.modules.daso import SMDMLiteV2, SCIMV2, build_hermitian_low_frequency_mask
from ultralytics.nn.tasks import OBBModel


class DASOTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(0)

    def test_mask_conjugate_symmetry(self):
        for height, width in ((16, 16), (15, 17)):
            mask = build_hermitian_low_frequency_mask(height, width, 0.25).bool()
            rows = (-torch.arange(height)) % height
            cols = (-torch.arange(width)) % width
            self.assertTrue(torch.equal(mask, mask[..., rows, :][..., cols]))

    def test_smdm_initialization_and_backward(self):
        module = SMDMLiteV2(16, diagnostic_dim=16, beta=0.25).eval()
        sar = torch.randn(2, 16, 8, 8, requires_grad=True)
        optical = torch.randn_like(sar, requires_grad=True)
        out_s, out_o = module(sar, optical)
        torch.testing.assert_close(out_s, module.proj_s(sar))
        torch.testing.assert_close(out_o, module.proj_o(optical))
        (out_s.square().mean() + out_o.square().mean()).backward()
        self.assertTrue(torch.isfinite(sar.grad).all())
        self.assertTrue(torch.isfinite(optical.grad).all())

    def test_scim_ordered_decomposition(self):
        base, reverse = torch.randn(2, 16, 8, 8), torch.randn(2, 16, 8, 8)
        common, d_reverse, d_base = SCIMV2.ordered_components(base, reverse)
        torch.testing.assert_close(common - d_reverse + d_base, base)
        torch.testing.assert_close(common + d_reverse - d_base, reverse)
        module = SCIMV2(16, state_ratio=1.0).eval()
        out = module(base, reverse)
        self.assertEqual(out.shape, base.shape)
        self.assertTrue(torch.isfinite(out).all())

    def test_all_configs(self):
        expected = {'baseline': 3713141, 'smdm': 7731829, 'scim': 4712565, 'daso-det': 8731253}
        for name, count in expected.items():
            with self.subTest(model=name):
                model = OBBModel(str(ROOT / 'ultralytics/cfg/models' / (name+'.yaml')),
                                 ch=6, nc=6, verbose=False).eval()
                self.assertEqual(sum(p.numel() for p in model.parameters()), count)
                with torch.inference_mode():
                    pred = model(torch.zeros(1, 6, 128, 128))[0]
                self.assertEqual(pred.shape[1], 11)  # xywh + six class scores + OBB angle
                self.assertTrue(torch.isfinite(pred).all())


if __name__ == '__main__':
    unittest.main()
