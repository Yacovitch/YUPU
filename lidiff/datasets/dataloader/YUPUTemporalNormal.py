import torch
from torch.utils.data import Dataset
from lidiff.utils.pcd_preprocess import point_set_to_coord_feats
from lidiff.utils.collations import point_set_to_sparse_grid_normal, point_set_to_sparse_yupu_normal
from lidiff.utils.pca_normals import estimate_normals_pca
from natsort import natsorted
import os
import numpy as np
import yaml

import warnings

warnings.filterwarnings('ignore')

class TemporalYUPUNormalSet(Dataset):
    def __init__(self, data_dir, seqs, split, resolution, num_points, max_range,
                 upsample_ratio=4, dataset_norm=False, std_axis_norm=False, grid=False,
                 synthetic_downsample=False, synthetic_downsample_method='random',
                 synthetic_normal_k=30):
        super().__init__()
        self.data_dir = data_dir

        self.resolution = resolution
        self.num_points = num_points
        self.upsample_ratio = upsample_ratio
        self.max_range = max_range
        self.grid = grid
        self.synthetic_downsample = bool(synthetic_downsample) and split == 'train'
        self.synthetic_downsample_method = synthetic_downsample_method
        self.synthetic_normal_k = int(synthetic_normal_k)

        if self.synthetic_downsample_method != 'random':
            raise ValueError(
                f"Unsupported synthetic downsampling method: {self.synthetic_downsample_method}. "
                "Available methods: random"
            )

        self.split = split
        self.seqs = seqs

        self.datapath_list()
        self.data_stats = {'mean': None, 'std': None}

        # Optional dataset normalization: load or compute stats
        if dataset_norm:
            stats_path = os.path.join('utils', f'data_stats_range_{int(self.max_range)}m.yml')
            if os.path.isfile(stats_path):
                stats = yaml.safe_load(open(stats_path))
                data_mean = np.array([stats['mean_axis']['x'], stats['mean_axis']['y'], stats['mean_axis']['z']])
                if std_axis_norm:
                    data_std = np.array([stats['std_axis']['x'], stats['std_axis']['y'], stats['std_axis']['z']])
                else:
                    data_std = np.array([stats['std'], stats['std'], stats['std']])
                self.data_stats = {
                    'mean': torch.tensor(data_mean),
                    'std': torch.tensor(data_std)
                }
            else:
                # Compute stats over GT frames (filtered by range on XY if provided)
                total = 0
                sum_xyz = np.zeros(3, dtype=np.float64)
                sum_sq_xyz = np.zeros(3, dtype=np.float64)
                for gt_path in self.points_gt_datapath:
                    pts = np.fromfile(gt_path, dtype=np.float32).reshape((-1,6))[:,:3]
                    if self.max_range is not None:
                        dist_xy = np.sqrt(np.sum(pts[:,:2]**2, axis=-1))
                        pts = pts[dist_xy < self.max_range]
                    n = pts.shape[0]
                    if n == 0:
                        continue
                    total += n
                    sum_xyz += pts.sum(axis=0)
                    sum_sq_xyz += (pts.astype(np.float64) ** 2).sum(axis=0)

                if total > 0:
                    mean_axis = sum_xyz / total
                    var_axis = np.maximum(sum_sq_xyz / total - mean_axis**2, 1e-12)
                    std_axis = np.sqrt(var_axis)
                    std_scalar = float(std_axis.mean())

                    stats = {
                        'mean_axis': {'x': float(mean_axis[0]), 'y': float(mean_axis[1]), 'z': float(mean_axis[2])},
                        'std_axis': {'x': float(std_axis[0]), 'y': float(std_axis[1]), 'z': float(std_axis[2])},
                        'std': std_scalar,
                    }
                    os.makedirs(os.path.dirname(stats_path), exist_ok=True)
                    with open(stats_path, 'w') as f:
                        yaml.safe_dump(stats, f)

                    if std_axis_norm:
                        data_std = std_axis
                    else:
                        data_std = np.array([std_scalar, std_scalar, std_scalar])
                    self.data_stats = {
                        'mean': torch.tensor(mean_axis.astype(np.float32)),
                        'std': torch.tensor(data_std.astype(np.float32))
                    }

        self.nr_data = len(self.points_datapath)

        print('The size of %s data is %d'%(self.split,len(self.points_datapath)))
        input_source = 'synthetic GT downsampling' if self.synthetic_downsample else 'real sparse scans'
        print(f'The {self.split} split uses {input_source}')

    def datapath_list(self):
        self.points_datapath = []
        self.points_gt_datapath = []
        self.normals_datapath = []

        for seq in self.seqs:
            point_seq_path = os.path.join(self.data_dir, 'dataset', 'sequences', seq)
            point_seq_bin = natsorted(os.listdir(os.path.join(point_seq_path, 'velodyne')))

            point_gt_path = os.path.join(self.data_dir, 'dataset', 'sequences_gt_x3', seq)
            point_gt_bin = natsorted(os.listdir(os.path.join(point_gt_path, 'velodyne')))
            point_gt_bin = [k for k in point_gt_bin if '.bin' in k]

            for file in point_gt_bin:
                self.points_datapath.append(os.path.join(point_seq_path, 'velodyne', file))
                self.points_gt_datapath.append(os.path.join(point_gt_path, 'velodyne', file))
                self.normals_datapath.append(os.path.join(point_seq_path, 'normals', file))

    def __getitem__(self, index):
        p_full = np.fromfile(self.points_gt_datapath[index], dtype=np.float32).reshape((-1,4))[:,:3]
        n_part = int(self.num_points / max(1, int(self.upsample_ratio)))

        if self.synthetic_downsample:
            if len(p_full) == 0:
                raise ValueError(f'Cannot downsample empty ground truth: {self.points_gt_datapath[index]}')
            # First form the exact ground-truth target used by the loss, then
            # downsample that same target to form the synthetic condition.
            if len(p_full) != self.num_points:
                full_idx = np.random.choice(
                    len(p_full), self.num_points, replace=len(p_full) < self.num_points
                )
                p_full = p_full[full_idx]
            sample_idx = np.random.choice(len(p_full), n_part, replace=len(p_full) < n_part)
            p_part = p_full[sample_idx]
            pmin, pmax = p_part.min(axis=0), p_part.max(axis=0)
            center_xy = (pmin[:2] + pmax[:2]) * 0.5
            margin = max(0.1 * float(np.linalg.norm(pmax - pmin)), 1.0)
            camera = np.array([center_xy[0], center_xy[1], pmax[2] + margin], dtype=np.float32)
            normals = estimate_normals_pca(p_part, k=self.synthetic_normal_k, orient_to_cam=camera)
        else:
            # Real sparse YUPU input and its precomputed normals.
            p_part = np.fromfile(self.points_datapath[index], dtype=np.float32).reshape((-1,4))[:,:3]
            normals = np.fromfile(self.normals_datapath[index], dtype=np.float32).reshape((-1,3))

        if self.grid:
            return point_set_to_sparse_grid_normal(
                p_full,
                p_part,
                self.num_points,
                n_part,
                self.resolution,
                self.points_datapath[index],
                p_mean=self.data_stats['mean'],
                p_std=self.data_stats['std'],
            )   
        else:
            return point_set_to_sparse_yupu_normal(
                p_full,
                p_part,
                self.num_points,
                n_part,
                self.resolution,
                normals,
                self.points_datapath[index],
                p_mean=self.data_stats['mean'],
                p_std=self.data_stats['std'],
            )

    def __len__(self):
        return self.nr_data
