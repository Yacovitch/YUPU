import torch
from torch.utils.data import Dataset
from lidiff.utils.pcd_preprocess import point_set_to_coord_feats
from lidiff.utils.collations import point_set_to_sparse_grid, point_set_to_sparse_yupu_normal
from natsort import natsorted
import os
import numpy as np

import warnings

warnings.filterwarnings('ignore')

class TemporalYUPUNormalSet(Dataset):
    def __init__(self, data_dir, seqs, split, resolution, num_points, max_range, dataset_norm=False, std_axis_norm=False, grid=False):
        super().__init__()
        self.data_dir = data_dir

        self.resolution = resolution
        self.num_points = num_points
        self.max_range = max_range
        self.grid = grid

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
        # YUPU: partial is already subsampled (~12000), GT is ~48000. No filtering.
        p_part = np.fromfile(self.points_datapath[index], dtype=np.float32).reshape((-1,4))[:,:3]
        p_full = np.fromfile(self.points_gt_datapath[index], dtype=np.float32).reshape((-1,4))[:,:3]
        normals = np.fromfile(self.normals_datapath[index], dtype=np.float32).reshape((-1,3))

        # Determine partial size target from config intent: num_points/upsample_ratio
        n_part = int(self.num_points / max(1, int(self.num_points / max(len(p_full), 1) * (len(p_part)/max(len(p_full),1)))))
        # In practice, for provided data, return as-is; downstream collate will handle sampling to exact sizes

        if self.grid:
            return point_set_to_sparse_grid(
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


