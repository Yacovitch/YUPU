# Reviewer experiments

## Synthetic-downsampling training

This experiment uses random downsampling of dense ground-truth points only for
the training split. PCA normals are estimated for those synthetic points. The
validation and test splits continue to use the real YUPU sparse scans and their
precomputed normals.

Run locally or inside the configured container:

```bash
python lidiff/train.py -c lidiff/config/yupu_normal_synthetic_config.yaml
```

On SLURM:

```bash
sbatch run_yupu_synthetic_cuda.sh
```

Evaluate the resulting checkpoint on the real sparse test split:

```bash
python lidiff/train.py \
  -c lidiff/config/yupu_normal_synthetic_config.yaml \
  --weights /path/to/checkpoint.ckpt \
  --test
```

The experiment checkpoints are written below:

```text
experiments/yupu_clip_normal_synthetic_downsample/default/version_*/checkpoints/
```

## FLOPs

Profile a trained checkpoint on one real sparse validation sample:

```bash
python lidiff/tools/profile_flops.py \
  -c lidiff/config/yupu_normal_synthetic_config.yaml \
  -ckpt /path/to/checkpoint.ckpt \
  -o experiments/yupu_clip_normal_synthetic_downsample/flops_report.json
```

The report includes counted FLOPs for conditional and unconditional forwards,
one classifier-free-guided denoising step, and the complete configured sampling
trajectory. PyTorch does not assign FLOPs to every custom MinkowskiEngine CUDA
kernel, so the JSON explicitly lists uncounted CUDA operators and labels the
numerical total as a lower bound.
