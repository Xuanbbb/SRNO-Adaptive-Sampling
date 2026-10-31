"""Fixed-budget, block-stratified sampling for SRNO training.

Scores affect allocation only. Every nonempty block receives at least one point,
and pixels are sampled uniformly WITHOUT replacement inside each block. Thus
area/count weights remain valid even though the block allocation is adaptive.
"""

import math

import torch
import torch.nn.functional as F


def full_hr_collate(samples):
    """Keep variable-size HR crops as a list (random scales differ per image)."""
    return {
        'inp': torch.stack([s['inp'] for s in samples]),
        'cell': torch.stack([s['cell'] for s in samples]),
        'gt': [s['gt'] for s in samples],
    }


def recovery_scores(prediction, block_grid):
    """Recovery-inspired indicator, not a certified SR error estimator.

Compare finite-difference RGB gradients with a local average of those gradients.
Coordinates span [-1, 1]; normalize derivatives by their coordinate spacing.
This regular-grid approximation is NOT R-PINN's triangular FEM estimator.
"""
    if prediction.ndim != 4 or min(prediction.shape[-2:]) < 2:
        raise ValueError('The preview must be BCHW with height and width >= 2.')
    h, w = prediction.shape[-2:]
    dy = (prediction[:, :, 1:, :] - prediction[:, :, :-1, :]) * (h / 2)
    dx = (prediction[:, :, :, 1:] - prediction[:, :, :, :-1]) * (w / 2)
    dy = F.pad(dy, (0, 0, 0, 1), mode='replicate')
    dx = F.pad(dx, (0, 1, 0, 0), mode='replicate')
    gradients = torch.cat([dy, dx], dim=1)
    recovered = F.avg_pool2d(F.pad(gradients, (1, 1, 1, 1), mode='replicate'), 3, stride=1)
    discrepancy = (gradients - recovered).square().mean(dim=1, keepdim=True).sqrt()
    return F.adaptive_avg_pool2d(discrepancy, block_grid).squeeze(1).detach()


def _allocate(capacities, scores, count, uniform_fraction, power):
    """Integer quotas with coverage, capacity limits, and exact total count."""
    capacities = capacities.to(dtype=torch.long)
    if not len(capacities) <= count <= int(capacities.sum()):
        raise ValueError('sample_q must be between the nonempty block count and HR pixel count.')
    area = capacities.to(dtype=torch.float64)
    score = scores.to(dtype=torch.float64)
    if not torch.isfinite(score).all() or (score < 0).any():
        raise ValueError('Difficulty scores must be finite and nonnegative.')
    # Scale before exponentiation, and use area-uniform allocation on flat images.
    score = score / score.max().clamp_min(1e-12)
    importance = area * score.pow(power)
    if importance.sum() == 0:
        importance = area
    mass = uniform_fraction * area / area.sum() + (1 - uniform_fraction) * importance / importance.sum()
    quota = torch.ones_like(capacities)
    remaining = count - len(capacities)
    while remaining:
        available = capacities - quota
        active_mass = mass * (available > 0)
        if active_mass.sum() == 0:
            active_mass = area * (available > 0)
        target = remaining * active_mass / active_mass.sum()
        addition = torch.minimum(target.floor().long(), available)
        if int(addition.sum()) == 0:
            # Largest remainder; random tie breaking avoids a top-left bias.
            jitter = torch.rand_like(target) * 1e-9
            order = (target + jitter).masked_fill(available == 0, -1).argsort(descending=True)
            addition[order[:min(remaining, int((available > 0).sum()))]] = 1
        quota += addition
        remaining -= int(addition.sum())
    return quota


@torch.no_grad()
def sample_batch(hr_images, block_scores, sample_q, uniform_fraction=0.5, power=1.0):
    """Return coords Bx1xNx2, RGB Bx3x1xN, weights Bx1x1xN.

Each weight is block_pixel_count / (image_pixel_count * block_sample_count).
Weights sum to one per image. The array layout is storage, not spatial adjacency.
"""
    if not isinstance(sample_q, int) or isinstance(sample_q, bool) or sample_q < 1:
        raise ValueError('sample_q must be a positive integer.')
    if not 0 <= uniform_fraction <= 1 or not math.isfinite(power) or power <= 0:
        raise ValueError('Require 0 <= uniform_fraction <= 1 and finite power > 0.')
    if block_scores.ndim != 3 or len(hr_images) != block_scores.shape[0]:
        raise ValueError('Expected one 2D block score map per HR image.')
    coords, values, weights = [], [], []
    for image, scores in zip(hr_images, block_scores):
        _, h, w = image.shape
        gh, gw = scores.shape
        if gh > h or gw > w:
            raise ValueError('block_grid must not exceed the HR crop dimensions.')
        # Quota decisions are small CPU operations; sample indices on image device.
        bounds = [(r * h // gh, (r + 1) * h // gh, c * w // gw, (c + 1) * w // gw)
                  for r in range(gh) for c in range(gw)]
        capacities = torch.tensor([(b-a)*(d-c) for a,b,c,d in bounds])
        quotas = _allocate(capacities, scores.detach().cpu().flatten(), sample_q,
                           uniform_fraction, power)
        indices, point_weights = [], []
        for (y0, y1, x0, x1), capacity, quota in zip(bounds, capacities.tolist(), quotas.tolist()):
            local = torch.randperm(capacity, device=image.device)[:quota]
            indices.append((local // (x1-x0) + y0) * w + local % (x1-x0) + x0)
            point_weights.append(image.new_full((quota,), capacity / (h*w*quota)))
        indices = torch.cat(indices)
        weight = torch.cat(point_weights)
        # Shuffle storage order so blocks do not acquire a positional layout.
        order = torch.randperm(sample_q, device=image.device)
        indices, weight = indices[order], weight[order]
        y = ((indices // w).to(image.dtype) + 0.5) * (2 / h) - 1
        x = ((indices % w).to(image.dtype) + 0.5) * (2 / w) - 1
        coords.append(torch.stack([y, x], dim=-1).unsqueeze(0))
        values.append(image.reshape(image.shape[0], -1)[:, indices].unsqueeze(1))
        weights.append(weight.reshape(1, 1, sample_q))
    return {'coord': torch.stack(coords), 'gt': torch.stack(values),
            'sample_weight': torch.stack(weights)}


def reconstruction_loss(prediction, target, sample_weight=None, mode='uniform'):
    """Area-weighted L1 estimates uniform-image error; sampled L1 is an ablation."""
    error = (prediction - target).abs()
    if mode == 'sampled' or sample_weight is None:
        return error.mean()
    if mode != 'uniform':
        raise ValueError('loss_measure must be uniform or sampled.')
    weight = sample_weight.flatten(2)
    weight = weight / weight.sum(dim=-1, keepdim=True)
    return (error.mean(dim=1, keepdim=True).flatten(2) * weight).sum(dim=-1).mean()
