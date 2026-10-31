import unittest

import torch
from torch.utils.data import DataLoader, Dataset

import models
from adaptive_sampling import (full_hr_collate, recovery_scores, sample_batch,
                               reconstruction_loss)
from datasets.wrappers import SRImplicitDownsampledFast
from models.galerkin import simple_attn


class SyntheticImages(Dataset):
    def __len__(self):
        return 2

    def __getitem__(self, index):
        return torch.rand(3, 40, 40)


class AdaptiveSamplingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(1)

    def test_fixed_budget_unique_coordinates_and_exact_area_integral(self):
        h, w = 17, 23
        image = torch.zeros(3, h, w)
        # Piecewise constant in the same 2x2 partition as the sampler.
        for r in range(2):
            for c in range(2):
                image[:, r*h//2:(r+1)*h//2, c*w//2:(c+1)*w//2] = 1 + 2*r+c
        selected = sample_batch([image], torch.tensor([[[0., 0.], [0., 10.]]]), 73)
        coord = selected['coord'][0, 0]
        self.assertEqual(coord.shape, (73, 2))
        self.assertEqual(torch.unique(coord, dim=0).shape[0], 73)
        weight = selected['sample_weight']
        self.assertTrue(torch.allclose(weight.sum(), torch.tensor(1.)))
        estimate = (selected['gt'] * weight).sum(-1).mean()
        self.assertTrue(torch.allclose(estimate, image.mean(), atol=1e-6))
        self.assertGreater(int(((coord[:, 0] >= -1+2*(h//2)/h) &
                                (coord[:, 1] >= -1+2*(w//2)/w)).sum()), 30)

    def test_capacity_flat_scores_and_invalid_budget(self):
        for count in (4, 17, 35):
            batch = sample_batch([torch.rand(3, 5, 7)], torch.zeros(1, 2, 2), count)
            self.assertEqual(batch['gt'].shape, (1, 3, 1, count))
            self.assertTrue(torch.isfinite(batch['sample_weight']).all())
        for count in (3, 36):
            with self.assertRaises(ValueError):
                sample_batch([torch.rand(3, 5, 7)], torch.zeros(1, 2, 2), count)
        saturated = sample_batch([torch.rand(3, 5, 7)],
                                 torch.tensor([[[0., 0.], [0., 1.]]]), 35,
                                 uniform_fraction=0)
        self.assertTrue(torch.isfinite(saturated['sample_weight']).all())

    def test_recovery_indicator_on_constant_and_edge(self):
        flat = torch.ones(1, 3, 16, 16)
        self.assertEqual(float(recovery_scores(flat, (4, 4)).max()), 0.)
        flat[:, :, :, 8:] = 0
        self.assertGreater(float(recovery_scores(flat, (4, 4)).max()), 0.)

    def test_attention_uniform_compatibility_and_permutation(self):
        layer = simple_attn(32, 4)
        x = torch.rand(2, 32, 1, 23)
        uniform = torch.ones(2, 1, 1, 23)
        self.assertTrue(torch.allclose(layer(x), layer(x, sample_weight=uniform), atol=1e-6))
        weights = torch.rand_like(uniform) + 0.1
        order = torch.randperm(23)
        actual = layer(x[..., order], sample_weight=weights[..., order])
        expected = layer(x, sample_weight=weights)[..., order]
        self.assertTrue(torch.allclose(actual, expected, atol=1e-5))

    def test_variable_scale_loader_preview_and_training_step(self):
        dataset = SRImplicitDownsampledFast(SyntheticImages(), inp_size=8,
                    scale_min=1, scale_max=3, augment=True, return_full_hr=True)
        batch = next(iter(DataLoader(dataset, batch_size=2, collate_fn=full_hr_collate)))
        model = models.make({'name': 'sronet', 'args': {
            'encoder_spec': {'name': 'edsr-baseline', 'args': {
                'no_upsampling': True, 'n_resblocks': 2}}, 'width': 32, 'blocks': 4}})
        from utils import make_coord
        inp = batch['inp'] * 2 - 1
        with torch.no_grad():
            preview = model(inp, make_coord((8, 8), flatten=False).unsqueeze(0).repeat(2, 1, 1, 1), batch['cell'])
            scores = recovery_scores(preview, (2, 2))
        selected = sample_batch(batch['gt'], scores, 32)
        pred = model(inp, selected['coord'], batch['cell'], selected['sample_weight'])
        loss = reconstruction_loss(pred, selected['gt'] * 2 - 1, selected['sample_weight'])
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        torch.optim.Adam(model.parameters(), lr=1e-4).step()
        # Existing 3-argument calls and state_dict format are retained.
        model.load_state_dict(model.state_dict(), strict=True)
        self.assertEqual(model(inp, selected['coord'], batch['cell']).shape, pred.shape)


if __name__ == '__main__':
    unittest.main()
