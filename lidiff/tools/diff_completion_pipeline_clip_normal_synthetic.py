"""Generate completions from synthetic sparse inputs sampled from YUPU GT.

This is the synthetic-input sister of ``diff_completion_pipeline_clip_normal``.
For each frame it first forms the configured 48,000-point ground-truth target,
randomly selects 12,000 conditioning points from that exact target, estimates
matching PCA normals, and runs the unchanged CLIP+normal completion model.
"""

import os
import time

import click
import numpy as np
import open3d as o3d
import torch
import tqdm
import yaml
from natsort import natsorted

from lidiff.tools.diff_completion_pipeline_clip_normal import DiffCompletion, load_pcd
from lidiff.utils.pca_normals import estimate_normals_pca


def random_downsample_ground_truth(points, full_size, sparse_size, seed):
    """Return an exact GT target and a random subset used as conditioning."""
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f'Expected ground truth with shape (N, 3+), got {points.shape}')
    points = points[:, :3]
    if len(points) == 0:
        raise ValueError('Cannot downsample an empty ground-truth point cloud')
    if full_size <= 0 or sparse_size <= 0:
        raise ValueError('full_size and sparse_size must be positive')
    if sparse_size > full_size:
        raise ValueError(f'sparse_size ({sparse_size}) cannot exceed full_size ({full_size})')

    rng = np.random.RandomState(seed)
    if len(points) == full_size:
        full = points.copy()
    else:
        full_idx = rng.choice(len(points), full_size, replace=len(points) < full_size)
        full = points[full_idx]

    sparse_idx = rng.choice(full_size, sparse_size, replace=False)
    return full, full[sparse_idx]


def estimate_input_normals(points, k):
    pmin, pmax = points.min(axis=0), points.max(axis=0)
    center_xy = (pmin[:2] + pmax[:2]) * 0.5
    margin = max(0.1 * float(np.linalg.norm(pmax - pmin)), 1.0)
    camera = np.array([center_xy[0], center_xy[1], pmax[2] + margin], dtype=np.float32)
    return estimate_normals_pca(points, k=k, orient_to_cam=camera)


class SyntheticGTCompletion(DiffCompletion):
    """DiffCompletion whose conditioning scan is sampled from ground truth."""

    def __init__(self, *args, random_seed=42, normal_k=30, **kwargs):
        super().__init__(*args, **kwargs)
        full_size = int(self.hparams['data']['num_points'])
        upsample_ratio = self.hparams['data'].get('upsample_ratio')
        if full_size != 48000 or upsample_ratio != 4:
            raise ValueError(
                'This experiment requires checkpoint hyperparameters '
                f'num_points=48000 and upsample_ratio=4; got {full_size} and {upsample_ratio}'
            )
        self.random_seed = int(random_seed)
        self.normal_k = int(normal_k)
        self.frame_index = 0
        self.last_full_target = None
        self.last_sparse_input = None

    def set_frame_index(self, frame_index):
        self.frame_index = int(frame_index)

    def preprocess_scan(self, ground_truth):
        full_size = int(self.hparams['data']['num_points'])
        upsample_ratio = int(self.hparams['data']['upsample_ratio'])
        if upsample_ratio <= 0 or full_size % upsample_ratio != 0:
            raise ValueError(
                f'num_points={full_size} must be divisible by upsample_ratio={upsample_ratio}'
            )
        sparse_size = full_size // upsample_ratio
        frame_seed = self.random_seed + self.frame_index
        full_target, sparse_input = random_downsample_ground_truth(
            ground_truth, full_size, sparse_size, frame_seed
        )
        sparse_normals = estimate_input_normals(sparse_input, self.normal_k)

        # Diffusion starts with one noisy point per output point. Repeating the
        # sparse condition matches the original pipeline while preserving the
        # 12,000 unique conditioning points used by CLIP and normal projection.
        repeated_input = np.tile(sparse_input, (upsample_ratio, 1))[:full_size]
        repeated_normals = np.tile(sparse_normals, (upsample_ratio, 1))[:full_size]

        self.last_full_target = full_target
        self.last_sparse_input = sparse_input
        scan = torch.from_numpy(repeated_input).float().cuda()[None, :, :]
        sub = torch.from_numpy(sparse_input).float().cuda()[None, :, :]
        normals = torch.from_numpy(repeated_normals).float().cuda()[None, :, :]
        return scan, sub, normals


def write_ply(path, points):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(np.asarray(points))
    return o3d.io.write_point_cloud(path, point_cloud)


@click.command()
@click.option('--diff', '-d', required=True, type=click.Path(exists=True), help='diffusion checkpoint')
@click.option('--refine', '-r', required=True, type=click.Path(exists=True), help='refinement checkpoint')
@click.option('--gt_path', required=True, type=click.Path(exists=True, file_okay=False),
              help='ground-truth velodyne directory containing matching .bin frames')
@click.option('--output_dir', '-o', required=True, type=click.Path(file_okay=False),
              help='directory for generated .ply scenes')
@click.option('--denoising_steps', '-T', type=int, default=50, show_default=True)
@click.option('--cond_weight', '-s', type=float, default=6.0, show_default=True)
@click.option('--random_seed', type=int, default=42, show_default=True,
              help='frame i uses random_seed + i')
@click.option('--normal_k', type=int, default=30, show_default=True)
@click.option('--save_reverse_diffusion', is_flag=True,
              help='save every fifth reverse-diffusion state')
@click.option('--save-refined/--no-save-refined', default=True, show_default=True)
def main(diff, refine, gt_path, output_dir, denoising_steps, cond_weight,
         random_seed, normal_k, save_reverse_diffusion, save_refined):
    os.makedirs(output_dir, exist_ok=True)
    completion = SyntheticGTCompletion(
        diff,
        refine,
        denoising_steps,
        cond_weight,
        random_seed=random_seed,
        normal_k=normal_k,
    )

    expected_full = int(completion.hparams['data']['num_points'])
    ratio = int(completion.hparams['data']['upsample_ratio'])
    expected_sparse = expected_full // ratio
    manifest = {
        'input_source': 'random subset of ground truth',
        'gt_path': os.path.abspath(gt_path),
        'diff_checkpoint': os.path.abspath(diff),
        'refine_checkpoint': os.path.abspath(refine),
        'full_points': expected_full,
        'sparse_points': expected_sparse,
        'upsample_ratio': ratio,
        'random_seed': random_seed,
        'normal_k': normal_k,
        'denoising_steps': denoising_steps,
        'conditioning_weight': cond_weight,
    }
    with open(os.path.join(output_dir, 'generation_config.yaml'), 'w') as config_file:
        yaml.safe_dump(manifest, config_file, sort_keys=False)

    frame_names = [
        name for name in natsorted(os.listdir(gt_path))
        if name.endswith(('.bin', '.ply')) and not name.startswith('._')
    ]
    for frame_index, frame_name in enumerate(tqdm.tqdm(frame_names)):
        stem = os.path.splitext(frame_name)[0]
        output_path = os.path.join(output_dir, f'{stem}.ply')
        if os.path.exists(output_path):
            print(f'Skipping {frame_name}: {output_path} already exists')
            continue

        ground_truth = load_pcd(os.path.join(gt_path, frame_name))
        completion.set_frame_index(frame_index)
        start = time.time()
        if save_reverse_diffusion:
            refined, diffusion, _, reverse_steps = completion.complete_scan(
                ground_truth, save_reverse_diffusion=True
            )
        else:
            refined, diffusion, _ = completion.complete_scan(
                ground_truth, save_reverse_diffusion=False
            )

        write_ply(output_path, diffusion)
        write_ply(os.path.join(output_dir, f'{stem}_synthetic_input.ply'), completion.last_sparse_input)
        if save_refined:
            write_ply(os.path.join(output_dir, f'{stem}_refine.ply'), refined)
        if save_reverse_diffusion:
            for step, points in enumerate(reverse_steps):
                if step % 5 == 0:
                    write_ply(os.path.join(output_dir, f'{stem}_reverse_{step}.ply'), points)
        print(
            f'{frame_name}: GT {len(completion.last_full_target)}, '
            f'synthetic input {len(completion.last_sparse_input)}, '
            f'generated {len(diffusion)}; took {time.time() - start:.2f}s'
        )


if __name__ == '__main__':
    main()
