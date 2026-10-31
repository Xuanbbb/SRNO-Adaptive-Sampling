# Fixed-budget adaptive sampling: version 1

This version changes **training sampling**, not the inference architecture or
output resolution. It does not implement hierarchical AMR or claim inference
speedups. The original config and three-argument model interface remain usable;
no parameters were added to checkpoints.

## Algorithm

1. Crop/augment HR and create LR exactly as in the upstream wrapper. Keep full
   HR targets in a variable-size list so independent random scales can coexist.
2. During the first 10 epochs, use area-uniform stratified sampling.
3. Afterwards, run a no-gradient 32x32 preview with the current model, using the
   requested target pixel cell size. Score 8x8 blocks using the difference between
   finite-difference RGB gradients and locally averaged gradients. This is a
   recovery-inspired heuristic, **not** the triangular FEM estimator or a
   certified estimate of SR reconstruction error.
4. Allocate exactly 4096 points. Combine 50% area-uniform and 50% difficulty mass,
   reserve one point per block, and respect available pixel capacities. All point
   coordinates are redrawn on each patch visit; there is no persistent background
   set and no dataset-level error cache. Uniform mass is not literally a fixed
   set of half the points after integer quota rounding and capacity handling.
5. Sample uniformly without replacement inside each block. For a block with
   `A` HR pixels, `n` samples, and `M` total image pixels, each point has weight
   `A/(M*n)`. These weights sum to one and enter **both attention layers**.
6. Default L1 uses the same area weights to approximate uniform-image error.
   `loss_measure: sampled` intentionally emphasizes oversampled regions instead.

The weighted sums correct the spatial measure. They do not make the nonlinear
network output an unbiased estimate of dense SRNO: each attention layer's features
also depend on the selected set. This distinction needs empirical evaluation.

The preview reads LR and current model predictions only; HR values are used for
supervision after point selection. Scoring happens every training batch after
warmup, rather than R-PINN's fixed-solution adaptive training rounds. The preview
adds a second encoder forward and full HR crops use more transfer memory, so
training acceleration must be measured, not assumed.

## Environment and running

Use Python 3.10+ with matching PyTorch/torchvision builds. Install the appropriate
CUDA wheels for your GPU before `python -m pip install -r requirements.txt`.
The existing `D:\EDSR\.conda\EDSR\python.exe` has torch/torchvision and can run
the tests below. Full training additionally needs PyYAML and tensorboardX; these
are listed in requirements.txt and were not installed during this change.
OpenCV and matplotlib are optional, needed only by `show_feature_map`.

Place DIV2K HR data under `data/DIV2K_train_HR` and `data/DIV2K_valid_HR`, or edit
both root paths in the configs. Images must support the configured HR crop size
(128 LR pixels times up to 4x). Training retains upstream's CUDA requirement.

```powershell
python train.py --config configs/train_edsr-sronet-adaptive.yaml --gpu 0 --name adaptive4096
python train.py --config configs/train_edsr-sronet-uniform4096.yaml --gpu 0 --name uniform4096
& 'D:\EDSR\.conda\EDSR\python.exe' -m unittest discover -s tests -v
```

Both new configs use batch size 8 and 0 loader workers for a conservative Windows
starting point. Adjust batch size to GPU memory and use the **same** value in
comparisons. Choose distinct run names; upstream prompts before reusing named
save directories (underscore-prefixed names are automatically replaced).

## Experiments

- Compare adaptive4096 against uniform4096 with equal epochs, batch size, data,
  and seeds. Uniform4096 uses the same stratified sampler with no preview.
- Compare with upstream config separately: it samples 16384 points and uses a
  larger batch size. Match batch size before attributing differences to sampling.
- Change `sample_q` to 9216 or 16384 for a budget sweep. It must be at least the
  number of blocks and no greater than every HR crop's pixel count.
- Measure validation PSNR, regional artifacts, total step time including preview
  and sampling, and peak GPU memory. No quality improvement has been established.
- Validation/test still use the upstream dense/chunked evaluation path. Its
  attention context changes with query chunks; keep `eval_bsize` fixed across runs.

Checks cover exact budgets, no duplicates, capacity handling, constant/edge
indicators, exact integration of blockwise constants, uniform-weight attention
compatibility, permutation equivariance, and synthetic preview/forward/backward.
