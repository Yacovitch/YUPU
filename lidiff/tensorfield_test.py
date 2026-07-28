import torch
import MinkowskiEngine as ME
import matplotlib.pyplot as plt
import numpy as np

# Function to project points to a 2D depth image
def project_points_to_image(points, image_size=(224, 224), depth_bins=112):
    """
    Projects 3D points onto a 2D image plane to generate a depth image.

    Args:
        points (torch.Tensor): 3D points of shape [N, 3].
        image_size (tuple): Resolution of the depth image (width, height).
        depth_bins (int): Number of depth levels.

    Returns:
        torch.Tensor: A depth image of shape [H, W].
    """
    width, height = image_size

    # Normalize spatial coordinates (x, y) to image dimensions
    x = ((points[:, 0] + 1) / 2 * (width - 1)).long()  # Normalize to [0, width)
    y = ((points[:, 1] + 1) / 2 * (height - 1)).long()  # Normalize to [0, height)

    # Normalize depth (z) to [0, depth_bins)
    z = ((points[:, 2] - points[:, 2].min()) / (points[:, 2].max() - points[:, 2].min()) * (depth_bins - 1)).long()

    # Initialize depth image
    depth_image = torch.zeros(height, width, dtype=torch.long, device=points.device)

    # Populate depth image (taking the nearest depth value per pixel)
    depth_image[y, x] = z
    return depth_image

# Function to load PLY file directly without using open3d
def load_ply_points(file_path):
    """
    Load point cloud data from a PLY file.
    
    Args:
        file_path (str): Path to the PLY file
        
    Returns:
        numpy.ndarray: Point cloud coordinates
    """
    with open(file_path, 'rb') as f:
        # Skip header until 'end_header'
        line = f.readline().decode('utf-8').strip()
        while line != 'end_header':
            line = f.readline().decode('utf-8').strip()
        
        # Read point data as binary
        points = np.fromfile(f, dtype=np.float32)
        
    # Reshape assuming each point has x, y, z coordinates
    # If the file includes other attributes (colors, normals), you'll need to adjust this
    points = points.reshape(-1, 3)
    
    return points

# Simple voxel downsampling
def voxel_downsample(points, voxel_size):
    """
    Downsample point cloud using voxel grid
    
    Args:
        points (numpy.ndarray): Input point cloud
        voxel_size (float): Voxel size for downsampling
        
    Returns:
        numpy.ndarray: Downsampled point cloud
    """
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
    for indices in voxels.values():
        downsampled_points.append(np.mean(points[indices], axis=0))
    
    return np.array(downsampled_points)

# Load outdoor point cloud scene
scene_path = "/nas2/jacob/LiDiff/lidiff/Datasets/test/000123.ply"  # Update with your file path
scene_points_np = load_ply_points(scene_path)
scene_points_np = voxel_downsample(scene_points_np, voxel_size=0.05)  # Downsample for faster processing
scene_points = torch.tensor(scene_points_np, dtype=torch.float32, device='cuda:1')

# Normalize scene points to [-1, 1] range
min_bounds = scene_points.min(dim=0)[0]
max_bounds = scene_points.max(dim=0)[0]
scene_points = 2 * (scene_points - min_bounds) / (max_bounds - min_bounds) - 1

# Add batch index to create x_feats
x_feats = torch.cat([torch.ones((scene_points.shape[0], 1), device='cuda:1'), scene_points], dim=1)  # Add batch index as the first column

# Quantized coordinates using MinkowskiEngine
x_coord = x_feats.clone()
resolution = 0.05
x_coord[:, 1:] = (x_coord[:, 1:] / resolution).round() * resolution  # Quantize spatial coordinates

# Create the TensorField
x_t = ME.TensorField(
    features=x_feats[:, 1:],  # Features
    coordinates=x_coord,  # Quantized coordinates
    quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
    minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
    device='cuda:1',
)

# Extract quantized coordinates from the TensorField
tensorfield_coords = x_t.coordinates[:, 1:4]  # Exclude the batch index

# Generate depth images
original_image = project_points_to_image(scene_points, image_size=(224, 224), depth_bins=112)
quantized_image = project_points_to_image(tensorfield_coords, image_size=(224, 224), depth_bins=112)

# Save the depth images
plt.imsave('original_scene_image.png', original_image.cpu().numpy(), cmap='viridis')
plt.imsave('quantized_scene_image.png', quantized_image.cpu().numpy(), cmap='viridis')

# Output file paths
print("Original scene image saved to: original_scene_image.png")
print("Quantized scene image saved to: quantized_scene_image.png")
