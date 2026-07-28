import torch
from torch.utils.data import Dataset
from lidiff.utils.collations import point_set_to_sparse_grid
from natsort import natsorted
import os
import numpy as np

import warnings

warnings.filterwarnings('ignore')

class TemporalYUPUGridSet(Dataset):
    def __init__(self, data_dir, seqs, split, resolution, num_points, upsample_ratio, max_range, dataset_norm=False, std_axis_norm=False):
        super().__init__()
        self.data_dir = data_dir

        self.resolution = resolution
        self.num_points = num_points
        self.max_range = max_range
        self.upsample_ratio = upsample_ratio

        self.split = split
        self.seqs = seqs

        self.datapath_list()
        self.data_stats = {'mean': None, 'std': None}

        self.nr_data = len(self.points_datapath)

        print('The size of %s data is %d'%(self.split,len(self.points_datapath)))

    def datapath_list(self):
        self.points_datapath = []
        self.points_gt_datapath = []

        for seq in self.seqs:
            point_seq_path = os.path.join(self.data_dir, 'dataset', 'sequences', seq)
            point_seq_bin = natsorted(os.listdir(os.path.join(point_seq_path, 'velodyne')))
            point_gt_path = os.path.join(self.data_dir, 'dataset', 'sequences_gt', seq)
            point_gt_bin = natsorted(os.listdir(os.path.join(point_gt_path, 'velodyne')))
            point_gt_bin = [k for k in point_gt_bin if '.bin' in k]

            for file in point_gt_bin:
                self.points_datapath.append(os.path.join(point_seq_path, 'velodyne', file))
                self.points_gt_datapath.append(os.path.join(point_gt_path, 'velodyne', file))

    def __getitem__(self, index):
        p_part = np.fromfile(self.points_datapath[index], dtype=np.float32).reshape((-1,4))[:,:3]
        p_full = np.fromfile(self.points_gt_datapath[index], dtype=np.float32).reshape((-1,6))[:,:3]

        # No filtering; directly map to sparse grid
        n_part = int(self.num_points / max(1, int(self.upsample_ratio)))

        return point_set_to_sparse_grid(
            p_full,
            p_part,
            self.num_points,
            n_part,
            self.resolution,
            self.points_datapath[index],
            p_mean=self.data_stats['mean'],
            p_std=self.data_stats['std'],
        )

    def __len__(self):
        return self.nr_data


