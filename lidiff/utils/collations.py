import numpy as np
import MinkowskiEngine as ME
import torch
import torch.nn.functional as F
import numpy as np
from lidiff.utils.pca_normals import estimate_normals_pca

# Replacement for open3d.utility.Vector3dVector
class Vector3dVector:
    def __init__(self, data):
        self.data = np.asarray(data)
    
    def __array__(self):
        return self.data

# Simple PointCloud class without open3d
class PointCloud:
    def __init__(self):
        self.points = None
        self.colors = None
        
    def voxel_down_sample(self, voxel_size):
        """Downsample point cloud using voxel grid"""
        if self.points is None:
            return self
        
        # Convert to numpy
        points = np.asarray(self.points)
        
        # Compute voxel indices for each point
        voxel_indices = np.floor(points / voxel_size).astype(int)
        
        # Create a dictionary to store points in each voxel
        voxels = {}
        for i, idx in enumerate(voxel_indices):
            idx_tuple = tuple(idx)
            if idx_tuple not in voxels:
                voxels[idx_tuple] = []
            voxels[idx_tuple].append(i)
        
        # Get the average point in each voxel
        downsampled_points = []
        downsampled_colors = []
        
        for indices in voxels.values():
            downsampled_points.append(np.mean(points[indices], axis=0))
            if self.colors is not None:
                colors = np.asarray(self.colors)
                downsampled_colors.append(np.mean(colors[indices], axis=0))
        
        result = PointCloud()
        result.points = Vector3dVector(np.array(downsampled_points))
        if self.colors is not None and downsampled_colors:
            result.colors = Vector3dVector(np.array(downsampled_colors))
        
        return result
    
    def farthest_point_down_sample(self, num_points):
        """Downsample point cloud using farthest point sampling"""
        if self.points is None:
            return self
        
        points = np.asarray(self.points)
        
        # If we don't have enough points, return the original point cloud
        if len(points) <= num_points:
            return self
        
        # Initialize with a random point
        sampled_indices = [np.random.randint(0, len(points))]
        sampled_pts = [points[sampled_indices[0]]]
        
        # Calculate distances to the initial point
        dists = np.sum((points - sampled_pts[0])**2, axis=1)
        
        # Iteratively select the farthest points
        for _ in range(1, num_points):
            # Select the farthest point from all sampled points
            new_idx = np.argmax(dists)
            sampled_indices.append(new_idx)
            sampled_pts.append(points[new_idx])
            
            # Update distances
            new_dists = np.sum((points - points[new_idx])**2, axis=1)
            dists = np.minimum(dists, new_dists)
        
        # Create a new point cloud with just the sampled points
        result = PointCloud()
        result.points = Vector3dVector(np.array(sampled_pts))
        
        if self.colors is not None:
            colors = np.asarray(self.colors)
            result.colors = Vector3dVector(colors[sampled_indices])
        
        return result

# VoxelGrid class without open3d
class VoxelGrid:
    def __init__(self, voxel_size=None):
        self.voxel_size = voxel_size
        self.voxels = set()
    
    @staticmethod
    def create_from_point_cloud(pcd, voxel_size):
        """Create a voxel grid from a point cloud"""
        points = np.asarray(pcd.points)
        grid = VoxelGrid(voxel_size)
        
        # Create voxels
        voxel_indices = np.floor(points / voxel_size).astype(int)
        for idx in voxel_indices:
            grid.voxels.add(tuple(idx))
        
        return grid
    
    def check_if_included(self, points_vector):
        """Check if points are within any voxel in the grid"""
        points = np.asarray(points_vector)
        voxel_indices = np.floor(points / self.voxel_size).astype(int)
        
        # Check each point's voxel against our set of voxels
        included = np.zeros(len(points), dtype=bool)
        for i, idx in enumerate(voxel_indices):
            if tuple(idx) in self.voxels:
                included[i] = True
        
        return included

def feats_to_coord(p_feats, resolution, mean, std):
    p_feats = p_feats.reshape(mean.shape[0],-1,3)
    p_coord = torch.round(p_feats / resolution)

    return p_coord.reshape(-1,3)

def normalize_pcd(points, mean, std):
    return (points - mean[:,None,:]) / std[:,None,:] if len(mean.shape) == 2 else (points - mean) / std

def unormalize_pcd(points, mean, std):
    return (points * std[:,None,:]) + mean[:,None,:] if len(mean.shape) == 2 else (points * std) + mean

def point_set_to_sparse_refine(p_full, p_part, n_full, n_part, resolution, filename):
    concat_full = np.ceil(n_full / p_full.shape[0])
    concat_part = np.ceil(n_part / p_part.shape[0])

    #if mode == 'diffusion':
    #p_full = p_full[torch.randperm(p_full.shape[0])]
    #p_part = p_part[torch.randperm(p_part.shape[0])]
    #elif mode == 'refine':
    p_full = p_full[torch.randperm(p_full.shape[0])]
    p_full = torch.tensor(p_full.repeat(concat_full, 0)[:n_full])   

    p_part = p_part[torch.randperm(p_part.shape[0])]
    p_part = torch.tensor(p_part.repeat(concat_part, 0)[:n_part])

    #p_feats = ME.utils.batched_coordinates([p_feats], dtype=torch.float32)[:2000]
    
    # after creating the voxel coordinates we normalize the floating coordinates towards mean=0 and std=1
    p_mean, p_std = p_full.mean(axis=0), p_full.std(axis=0)

    return [p_full, p_mean, p_std, p_part, filename]

def point_set_to_sparse(p_full, p_part, n_full, n_part, resolution, filename, p_mean=None, p_std=None):
    concat_part = np.ceil(n_part / p_part.shape[0]) 
    p_part = p_part.repeat(concat_part, 0)
    pcd_part = PointCloud()
    pcd_part.points = Vector3dVector(p_part)
    viewpoint_grid = VoxelGrid.create_from_point_cloud(pcd_part, voxel_size=10.)
    pcd_part = pcd_part.farthest_point_down_sample(n_part)
    p_part = torch.tensor(np.array(pcd_part.points))
    
    in_viewpoint = viewpoint_grid.check_if_included(Vector3dVector(p_full))
    p_full = p_full[in_viewpoint] 
    concat_full = np.ceil(n_full / p_full.shape[0])

    p_full = p_full[torch.randperm(p_full.shape[0])]
    p_full = p_full.repeat(concat_full, 0)[:n_full]

    p_full = torch.tensor(p_full)
    
    # after creating the voxel coordinates we normalize the floating coordinates towards mean=0 and std=1
    p_mean = p_full.mean(axis=0) if p_mean is None else p_mean
    p_std = p_full.std(axis=0) if p_std is None else p_std

    return [p_full, p_mean, p_std, p_part, filename]
    
def point_set_to_sparse_normal(p_full, p_part, n_full, n_part, resolution, filename, p_mean=None, p_std=None):
    concat_part = np.ceil(n_part / p_part.shape[0]) 
    p_part = p_part.repeat(concat_part, 0)
    pcd_part = PointCloud()
    pcd_part.points = Vector3dVector(p_part)
    viewpoint_grid = VoxelGrid.create_from_point_cloud(pcd_part, voxel_size=10.)
    pcd_part = pcd_part.farthest_point_down_sample(n_part)
    
    pmin = p_part.min(axis=0)
    pmax = p_part.max(axis=0)
    center_xy = (pmin[:2] + pmax[:2]) * 0.5
    diag = float(np.linalg.norm(pmax - pmin))
    margin = 0.1 * diag if np.isfinite(diag) and diag > 0 else 1.0
    cam_z = float(pmax[2] + margin)
    cam = np.array([center_xy[0], center_xy[1], cam_z], dtype=np.float32)
    normals = estimate_normals_pca(p_part, k=30, orient_to_cam=cam)
    
    p_part = torch.tensor(np.array(pcd_part.points))
    normals = torch.tensor(normals)
    
    in_viewpoint = viewpoint_grid.check_if_included(Vector3dVector(p_full))
    p_full = p_full[in_viewpoint] 
    concat_full = np.ceil(n_full / p_full.shape[0])

    p_full = p_full[torch.randperm(p_full.shape[0])]
    p_full = p_full.repeat(concat_full, 0)[:n_full]

    p_full = torch.tensor(p_full)
    
    # after creating the voxel coordinates we normalize the floating coordinates towards mean=0 and std=1
    p_mean = p_full.mean(axis=0) if p_mean is None else p_mean
    p_std = p_full.std(axis=0) if p_std is None else p_std

    return [p_full, p_mean, p_std, p_part, normals, filename]

def point_set_to_sparse_yupu(p_full, p_part, n_full, n_part, resolution, filename, p_mean=None, p_std=None):
    p_full = _sample_to_size(p_full, n_full, 'full point cloud')
    p_part = _sample_to_size(p_part, n_part, 'partial point cloud')

    p_part = torch.as_tensor(p_part)
    p_full = torch.as_tensor(p_full)
    
    # after creating the voxel coordinates we normalize the floating coordinates towards mean=0 and std=1
    p_mean = p_full.mean(axis=0) if p_mean is None else p_mean
    p_std = p_full.std(axis=0) if p_std is None else p_std

    return [p_full, p_mean, p_std, p_part, filename]

def point_set_to_sparse_yupu_normal(p_full, p_part, n_full, n_part, resolution, normals, filename, p_mean=None, p_std=None):
    if len(p_part) != len(normals):
        raise ValueError(
            f'Partial point cloud and normals must have the same length; '
            f'got {len(p_part)} and {len(normals)} for {filename}'
        )

    p_full = _sample_to_size(p_full, n_full, 'full point cloud')
    part_indices = _sample_indices(len(p_part), n_part, 'partial point cloud')
    p_part = p_part[part_indices]
    normals = normals[part_indices]

    p_part = torch.as_tensor(p_part)
    normals = torch.as_tensor(normals)
    p_full = torch.as_tensor(p_full)
    
    # after creating the voxel coordinates we normalize the floating coordinates towards mean=0 and std=1
    p_mean = p_full.mean(axis=0) if p_mean is None else p_mean
    p_std = p_full.std(axis=0) if p_std is None else p_std

    return [p_full, p_mean, p_std, p_part, normals, filename]


def _sample_indices(current_size, target_size, name):
    """Return indices that make a point set exactly the configured size."""
    if current_size == 0:
        raise ValueError(f'Cannot sample an empty {name}')
    if target_size <= 0:
        raise ValueError(f'Target size for {name} must be positive, got {target_size}')
    if current_size == target_size:
        return np.arange(current_size)
    return np.random.choice(current_size, target_size, replace=current_size < target_size)


def _sample_to_size(points, target_size, name):
    return points[_sample_indices(len(points), target_size, name)]

def random_sub_sampling(points, num_output, verbose=0):
    num_input = np.shape(points)[0]
    #num_output = num_input // sub_ratio
    idx = np.random.choice(num_input, num_output)
    return points[idx]

def point_set_to_sparse_grid(p_full, p_part, n_full, n_part, resolution, filename, p_mean=None, p_std=None):
    concat_part = np.ceil(n_part / p_part.shape[0])

    p_part = random_sub_sampling(p_part, n_part)
    p_part = p_part[torch.randperm(p_part.shape[0])]
    p_part = p_part.repeat(concat_part, 0)

    p_part = torch.tensor(p_part)
    
    
    concat_full = np.ceil(n_full / p_full.shape[0])
    
    p_full = random_sub_sampling(p_full, n_full)
    p_full = p_full[torch.randperm(p_full.shape[0])]
    p_full = p_full.repeat(concat_full, 0)[:n_full]

    p_full = torch.tensor(p_full)
    
    # after creating the voxel coordinates we normalize the floating coordinates towards mean=0 and std=1
    p_mean = p_full.mean(axis=0) if p_mean is None else p_mean
    p_std = p_full.std(axis=0) if p_std is None else p_std

    return [p_full, p_mean, p_std, p_part, filename]


def point_set_to_sparse_grid_normal(p_full, p_part, n_full, n_part, resolution, filename, p_mean=None, p_std=None):
    concat_part = np.ceil(n_part / p_part.shape[0])

    p_part = random_sub_sampling(p_part, n_part)
    p_part = p_part[torch.randperm(p_part.shape[0])]
    p_part = p_part.repeat(concat_part, 0)
    
    pmin = p_part.min(axis=0)
    pmax = p_part.max(axis=0)
    center_xy = (pmin[:2] + pmax[:2]) * 0.5
    diag = float(np.linalg.norm(pmax - pmin))
    margin = 0.1 * diag if np.isfinite(diag) and diag > 0 else 1.0
    cam_z = float(pmax[2] + margin)
    cam = np.array([center_xy[0], center_xy[1], cam_z], dtype=np.float32)
    normals = estimate_normals_pca(p_part, k=30, orient_to_cam=cam)

    p_part = torch.tensor(p_part)
    normals = torch.tensor(normals)
    
    concat_full = np.ceil(n_full / p_full.shape[0])
    
    p_full = random_sub_sampling(p_full, n_full)
    p_full = p_full[torch.randperm(p_full.shape[0])]
    p_full = p_full.repeat(concat_full, 0)[:n_full]

    p_full = torch.tensor(p_full)
    
    # after creating the voxel coordinates we normalize the floating coordinates towards mean=0 and std=1
    p_mean = p_full.mean(axis=0) if p_mean is None else p_mean
    p_std = p_full.std(axis=0) if p_std is None else p_std

    return [p_full, p_mean, p_std, p_part, normals, filename]

def numpy_to_sparse_tensor(p_coord, p_feats, p_label=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    p_coord = ME.utils.batched_coordinates(p_coord, dtype=torch.float32)
    p_feats = torch.vstack(p_feats).float()

    if p_label is not None:
        p_label = ME.utils.batched_coordinates(p_label, device=torch.device('cpu')).numpy()
    
        return ME.SparseTensor(
                features=p_feats,
                coordinates=p_coord,
                device=device,
            ), p_label

    return ME.SparseTensor(
                features=p_feats,
                coordinates=p_coord,
                device=device,
            )

class SparseSegmentCollation:
    def __init__(self, mode='diffusion'):
        self.mode = mode
        return

    def __call__(self, data):
        # "transpose" the  batch(pt, ptn) to batch(pt), batch(ptn)
        batch = list(zip(*data))

        return {'pcd_full': torch.stack(batch[0]).float(),
            'mean': torch.stack(batch[1]).float(),
            'std': torch.stack(batch[2]).float(),
            'pcd_part' if self.mode == 'diffusion' else 'pcd_noise': torch.stack(batch[3]).float(),
            'filename': batch[4],
        }

class SparseSegmentCollationNormal:
    def __init__(self, mode='diffusion'):
        self.mode = mode
        return

    def __call__(self, data):
        # "transpose" the  batch(pt, ptn) to batch(pt), batch(ptn)
        batch = list(zip(*data))

        return {'pcd_full': torch.stack(batch[0]).float(),
            'mean': torch.stack(batch[1]).float(),
            'std': torch.stack(batch[2]).float(),
            'pcd_part' if self.mode == 'diffusion' else 'pcd_noise': torch.stack(batch[3]).float(),
            'normals': torch.stack(batch[4]).float(),
            'filename': batch[5],
        }
