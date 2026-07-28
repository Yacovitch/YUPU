import torch
import torch.nn as nn
import torch.nn.functional as F
import lidiff.models.minkunet as minknet
import numpy as np
import MinkowskiEngine as ME
from lidiff.utils.scheduling import beta_func
from tqdm import tqdm
from os import makedirs
from pytorch3d.loss import chamfer_distance

from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import LightningDataModule
from lidiff.utils.collations import *
from lidiff.utils.metrics import ChamferDistance, PrecisionRecall

# Simple PointCloud class for visualization without open3d
class SimplePointCloud:
    def __init__(self):
        self.points = None
        self.colors = None
        self.normals = None
    
    def set_points(self, points):
        self.points = points
        return self
    
    def paint_uniform_color(self, color):
        """Paint the point cloud with a uniform color"""
        if self.points is None:
            return self
        
        self.colors = np.ones((len(self.points), 3)) * color
        return self
    
    def estimate_normals(self, k=20):
        """Estimate normals using PCA on k nearest neighbors"""
        if self.points is None or len(self.points) < k:
            return self
            
        # Simple placeholder - in a real implementation, you would compute
        # normals using nearest neighbors and PCA
        self.normals = np.zeros_like(self.points)
        return self
    
    def save_ply(self, filename):
        """Save point cloud to PLY file"""
        if self.points is None:
            return
        
        with open(filename, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(self.points)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            
            if self.normals is not None:
                f.write("property float nx\n")
                f.write("property float ny\n")
                f.write("property float nz\n")
                
            if self.colors is not None:
                f.write("property uchar red\n")
                f.write("property uchar green\n")
                f.write("property uchar blue\n")
            
            f.write("end_header\n")
            
            for i in range(len(self.points)):
                line = f"{self.points[i, 0]} {self.points[i, 1]} {self.points[i, 2]}"
                
                if self.normals is not None:
                    line += f" {self.normals[i, 0]} {self.normals[i, 1]} {self.normals[i, 2]}"
                    
                if self.colors is not None:
                    r = int(self.colors[i, 0] * 255)
                    g = int(self.colors[i, 1] * 255)
                    b = int(self.colors[i, 2] * 255)
                    line += f" {r} {g} {b}"
                
                f.write(line + "\n")

class RefineDiffusion(LightningModule):
    def __init__(self, hparams:dict, data_module: LightningDataModule = None):
        super().__init__()
        # name you hyperparameter hparams, then it will be saved automagically.
        self.save_hyperparameters(hparams)
        self.data_module = data_module

        # learn N offsets per point: out_channel is 3 * N
        self.model_refine = minknet.MinkUNet(in_channels=3, out_channels=3*self.hparams['train']['up_factor'])

        n_part = int(self.hparams['data']['num_points'] / self.hparams['data']['scan_window'])
        self.chamfer_distance = ChamferDistance()
        self.precision_recall = PrecisionRecall(0.001,0.01,100)
    
    def points_to_tensor(self, x_feats, mean, std):
        x_feats = ME.utils.batched_coordinates(list(x_feats[:]), dtype=torch.float32, device=self.device)

        x_coord = x_feats.clone()
        x_coord[:,1:] = feats_to_coord(x_feats[:,1:], self.hparams['data']['resolution'], mean, std)

        x_t = ME.TensorField(
            features=x_feats[:,1:],
            coordinates=x_coord,
            quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
            minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
            device=self.device,
        )

        torch.cuda.empty_cache()

        return x_t

    def forward_refine(self, x):
        return self.model_refine(x)

    def training_step(self, batch, batch_idx):
        x_feats = ME.utils.batched_coordinates(list(batch['pcd_noise']), dtype=torch.float32, device=self.device)
        x_coord = x_feats.clone()
        x_coord = torch.round(x_feats / self.hparams['data']['resolution'])

        x_feats = x_feats[:,1:]

        x_t = ME.TensorField(
            features=x_feats,
            coordinates=x_coord,
            quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
            minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
            device=self.device,
        )

        offset = self.forward_refine(x_t).reshape(-1,self.hparams['train']['up_factor'],3)
        refine_upsample_pcd = x_feats[:,None,:] + offset
        refine_upsample_pcd = refine_upsample_pcd.reshape(batch['pcd_full'].shape[0],-1,3)

        loss, _ = chamfer_distance(refine_upsample_pcd, torch.tensor(batch['pcd_full']))
        self.log('train/cd_loss', loss)
        torch.cuda.empty_cache()

        return loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            x_feats = ME.utils.batched_coordinates(list(batch['pcd_noise']), dtype=torch.float32, device=self.device)
            x_coord = x_feats.clone()
            x_coord = torch.round(x_feats / self.hparams['data']['resolution'])
    
            x_feats = x_feats[:,1:]
    
            x_t = ME.TensorField(
                features=x_feats,
                coordinates=x_coord,
                quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
                minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
                device=self.device,
            )
    
            offset = self.forward_refine(x_t).reshape(-1,self.hparams['train']['up_factor'],3)
            refine_upsample_pcd = x_feats[:,None,:] + offset
            refine_upsample_pcd = refine_upsample_pcd.reshape(batch['pcd_full'].shape[0],-1,3)
    
            loss, _ = chamfer_distance(refine_upsample_pcd, torch.tensor(batch['pcd_full']))
            self.log('val/cd_loss', loss)
            torch.cuda.empty_cache()
    
            return loss

    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            x_feats = ME.utils.batched_coordinates(list(batch['pcd_noise']), dtype=torch.float32, device=self.device)
            x_coord = x_feats.clone()
            x_coord = torch.round(x_feats / self.hparams['data']['resolution'])
    
            x_feats = x_feats[:,1:]
    
            x_t = ME.TensorField(
                features=x_feats,
                coordinates=x_coord,
                quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
                minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
                device=self.device,
            )
    
            offset = self.forward_refine(x_t).reshape(-1,self.hparams['train']['up_factor'],3)
            refine_pcd = x_feats[:,None,:] + offset
            refine_pcd = refine_pcd.reshape(batch['pcd_full'].shape[0],-1,3)

            pcd_refine = SimplePointCloud()
            pcd_refine.set_points(refine_pcd[0].cpu().numpy())
            pcd_refine.paint_uniform_color([1.,.2,.2])
            pcd_refine.estimate_normals()
            # Save to a file instead of visualization
            pcd_refine.save_ply(f"refine_output_{batch_idx}.ply")
            print(f"Saved point cloud to refine_output_{batch_idx}.ply")
    
            loss, _ = chamfer_distance(refine_pcd, torch.tensor(batch['pcd_full']))
            self.log('test/cd_loss', loss)
            torch.cuda.empty_cache()

            return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams['train']['lr'], betas=(0.9, 0.999))

        return optimizer

#######################################
# Modules
#######################################
