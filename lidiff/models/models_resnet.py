import torch
import torch.nn as nn
import torch.nn.functional as F
import lidiff.models.minkunet as minknet
import numpy as np
import MinkowskiEngine as ME
from lidiff.utils.scheduling import beta_func
from tqdm import tqdm
from os import makedirs, path
from scipy.spatial import cKDTree
import torchvision.models as models
import gc

from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import LightningDataModule
from lidiff.utils.collations import *
from lidiff.utils.metrics import ChamferDistance, PrecisionRecall
from diffusers import DPMSolverMultistepScheduler

import sys

from lidiff.models.img_feature_extractor import Extractor, Extractor_img
from lidiff.models.unprojection import vanilla_upprojection

import matplotlib.pyplot as plt

# Simple PointCloud class for visualization without open3d
class SimplePointCloud:
    def __init__(self, points=None):
        self.points = points
        self.colors = None
    
    def set_points(self, points):
        self.points = points
        return self
    
    def set_colors(self, colors):
        self.colors = colors
        return self
    
    def compute_point_cloud_distance(self, other_pcd):
        """Compute the distance from each point in this point cloud to the nearest point in the other point cloud"""
        if self.points is None or other_pcd.points is None:
            return np.array([])
            
        # Convert points to numpy array if they're not already
        if isinstance(self.points, torch.Tensor):
            self.points = self.points.detach().cpu().numpy()
        if isinstance(other_pcd.points, torch.Tensor):
            other_pcd.points = other_pcd.points.detach().cpu().numpy()
            
        # Ensure points are 2D arrays
        if len(self.points.shape) == 1:
            self.points = self.points.reshape(1, -1)
        if len(other_pcd.points.shape) == 1:
            other_pcd.points = other_pcd.points.reshape(1, -1)
            
        tree = cKDTree(other_pcd.points)
        distances, _ = tree.query(self.points)
        return distances
    
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
            
            if self.colors is not None:
                f.write("property uchar red\n")
                f.write("property uchar green\n")
                f.write("property uchar blue\n")
            
            f.write("end_header\n")
            
            for i in range(len(self.points)):
                line = f"{self.points[i, 0]} {self.points[i, 1]} {self.points[i, 2]}"
                
                if self.colors is not None:
                    r = int(self.colors[i, 0] * 255)
                    g = int(self.colors[i, 1] * 255)
                    b = int(self.colors[i, 2] * 255)
                    line += f" {r} {g} {b}"
                
                f.write(line + "\n")

class ImageResNet(nn.Module):
    def __init__(self, device):
        super(ImageResNet, self).__init__()
        # Load pretrained ResNet-50
        resnet = models.resnet50(pretrained=True)
        # Remove the final classification layer
        self.model = nn.Sequential(*list(resnet.children())[:-1])
        self.model = self.model.to(device)
        self.model.eval()
        
    def forward(self, image):
        with torch.no_grad():
            features = self.model(image)
            # Reshape from [B, 2048, 1, 1] to [B, 2048]
            features = features.squeeze(-1).squeeze(-1)
            # Normalize features
            features = F.normalize(features, p=2, dim=-1)
        return features

class DiffusionPoints(LightningModule):
    def __init__(self, hparams:dict, data_module: LightningDataModule = None, config_path=None):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.data_module = data_module
        self.config_path = config_path
        
        # Enable mixed precision training
        self.automatic_optimization = False  # We'll handle optimization manually for mixed precision

        # alphas and betas
        if self.hparams['diff']['beta_func'] == 'cosine':
            self.betas = beta_func[self.hparams['diff']['beta_func']](self.hparams['diff']['t_steps'])
        else:
            self.betas = beta_func[self.hparams['diff']['beta_func']](
                    self.hparams['diff']['t_steps'],
                    self.hparams['diff']['beta_start'],
                    self.hparams['diff']['beta_end'],
            )

        self.t_steps = self.hparams['diff']['t_steps']
        self.s_steps = self.hparams['diff']['s_steps']
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.tensor(
            np.cumprod(self.alphas, axis=0), dtype=torch.float32, device=torch.device('cuda')
        )

        self.alphas_cumprod_prev = torch.tensor(
            np.append(1., self.alphas_cumprod[:-1].cpu().numpy()), dtype=torch.float32, device=torch.device('cuda')
        )

        self.betas = torch.tensor(self.betas, device=torch.device('cuda'))
        self.alphas = torch.tensor(self.alphas, device=torch.device('cuda'))

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1. - self.alphas_cumprod) 
        self.sqrt_recip_alphas = torch.sqrt(1. / self.alphas)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod - 1.)

        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.sqrt_posterior_variance = torch.sqrt(self.posterior_variance)
        self.posterior_log_var = torch.log(
            torch.max(self.posterior_variance, 1e-20 * torch.ones_like(self.posterior_variance))
        )

        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1. - self.alphas_cumprod)
        
        # for fast sampling
        self.dpm_scheduler = DPMSolverMultistepScheduler(
                num_train_timesteps=self.t_steps,
                beta_start=self.hparams['diff']['beta_start'],
                beta_end=self.hparams['diff']['beta_end'],
                beta_schedule='linear',
                algorithm_type='sde-dpmsolver++',
                solver_order=2,
        )
        self.dpm_scheduler.set_timesteps(self.s_steps)
        self.scheduler_to_cuda()

        self.partial_enc = minknet.MinkGlobalEnc(in_channels=3, out_channels=self.hparams['model']['out_dim'])
        self.partial_enc_img = minknet.MinkowskiEncoder256(in_channels=2048, feature_dim=256)  # Changed to 2048 for ResNet-50
        
        # Initialize ResNet-50 model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.resnet_model = ImageResNet(device=device)
        # Load CLIP model
        self.img_extractor = Extractor_img(
            device=device,
            n_points=self.hparams['data']['num_points'] // self.hparams['data'].get('upsample_ratio', 10),
        )
        
        self.model = minknet.MinkUNetDiffClip(in_channels=3, out_channels=self.hparams['model']['out_dim'])

        self.chamfer_distance = ChamferDistance()
        self.precision_recall = PrecisionRecall(self.hparams['data']['resolution'],2*self.hparams['data']['resolution'],100)

        self.w_uncond = self.hparams['train']['uncond_w']

    def cleanup_cache(self):
        """Comprehensive cache cleanup to free GPU memory"""
        # Clear PyTorch cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Force garbage collection
        gc.collect()
        
        # Clear MinkowskiEngine cache if available
        if hasattr(ME, 'clear_global_coordinate_manager'):
            ME.clear_global_coordinate_manager()

    def scheduler_to_cuda(self):
        self.dpm_scheduler.timesteps = self.dpm_scheduler.timesteps.cuda()
        self.dpm_scheduler.betas = self.dpm_scheduler.betas.cuda()
        self.dpm_scheduler.alphas = self.dpm_scheduler.alphas.cuda()
        self.dpm_scheduler.alphas_cumprod = self.dpm_scheduler.alphas_cumprod.cuda()
        self.dpm_scheduler.alpha_t = self.dpm_scheduler.alpha_t.cuda()
        self.dpm_scheduler.sigma_t = self.dpm_scheduler.sigma_t.cuda()
        self.dpm_scheduler.lambda_t = self.dpm_scheduler.lambda_t.cuda()
        self.dpm_scheduler.sigmas = self.dpm_scheduler.sigmas.cuda()

    def q_sample(self, x, t, noise):
        return self.sqrt_alphas_cumprod[t][:,None,None].cuda() * x + \
                self.sqrt_one_minus_alphas_cumprod[t][:,None,None].cuda() * noise

    def classfree_forward(self, x_t, x_cond, x_uncond, t, pcd_part, mean, std):
        x_t_sparse = x_t.sparse()
        print(f'cond  | x_cond  : {x_cond.shape}, pcd_part: {pcd_part.shape}')
        print(f'uncond| x_uncond: {x_uncond.shape}, pcd_part: {torch.zeros_like(pcd_part).shape}')
        x_cond = self.forward(x_t, x_t_sparse, x_cond, t, pcd_part, mean, std)            
        x_uncond = self.forward(x_t, x_t_sparse, x_uncond, t, torch.zeros_like(pcd_part), torch.zeros_like(mean), torch.zeros_like(std))

        return x_uncond + self.w_uncond * (x_cond - x_uncond)

    def visualize_step_t(self, x_t, gt_pts, pcd, pcd_mean, pcd_std, pidx=0):
        points = x_t.F.detach().cpu().numpy()
        points = points.reshape(gt_pts.shape[0],-1,3)
        obj_mean = pcd_mean[pidx][0].detach().cpu().numpy()
        points = np.concatenate((points[pidx], gt_pts[pidx]), axis=0)

        dist_pts = np.sqrt(np.sum((points - obj_mean)**2, axis=-1))
        dist_idx = dist_pts < self.hparams['data']['max_range']

        full_pcd = len(points) - len(gt_pts[pidx])
        print(f'\n[{dist_idx.sum() - full_pcd}|{dist_idx.shape[0] - full_pcd }] points inside margin...')

        # Set points and colors for visualization
        pcd.set_points(points[dist_idx])
       
        colors = np.ones((len(points), 3)) * .5
        colors[:len(gt_pts[0])] = [1.,.3,.3]
        colors[-len(gt_pts[0]):] = [.3,1.,.3]
        pcd.set_colors(colors[dist_idx])

    def reset_partial_pcd(self, x_part, x_uncond, x_mean, x_std):
        x_part = self.points_to_tensor(x_part.F.reshape(x_mean.shape[0],-1,3).detach(), x_mean, x_std)
        x_uncond = self.points_to_tensor(
                torch.zeros_like(x_part.F.reshape(x_mean.shape[0],-1,3)), torch.zeros_like(x_mean), torch.zeros_like(x_std)
        )

        return x_part, x_uncond

    def p_sample_loop(self, x_init, x_t, x_cond, x_uncond, gt_pts, x_part_batch, x_mean, x_std):
        pcd = SimplePointCloud()
        self.scheduler_to_cuda()

        for t in tqdm(range(len(self.dpm_scheduler.timesteps))):
            t = torch.ones(gt_pts.shape[0]).cuda().long() * self.dpm_scheduler.timesteps[t].cuda()
            
            noise_t = self.classfree_forward(x_t, x_cond, x_uncond, t, x_part_batch, x_mean, x_std)
            input_noise = x_t.F.reshape(t.shape[0],-1,3) - x_init
            x_t = x_init + self.dpm_scheduler.step(noise_t, t[0], input_noise)['prev_sample']
            x_t = self.points_to_tensor(x_t, x_mean, x_std)

            # this is needed otherwise minkEngine will keep "stacking" coords maps over the x_part and x_uncond
            # i.e. memory leak
            x_cond, x_uncond = self.reset_partial_pcd(x_cond, x_uncond, x_mean, x_std)
            self.cleanup_cache()

        makedirs(f'{self.logger.log_dir}/generated_pcd/', exist_ok=True)

        return x_t

    def p_losses(self, y, noise):
        return F.mse_loss(y, noise)

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

        # Clean up cache after tensor creation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return x_t

    def points_to_tensor_with_features(self, x_feats, features, mean, std):
        features = features.squeeze(0)
        coordinates = ME.utils.batched_coordinates(list(x_feats[:]), dtype=torch.float32, device=self.device)
        normalized_coordinates = coordinates.clone()
        normalized_coordinates[:, 1:] = feats_to_coord(coordinates[:, 1:], self.hparams['data']['resolution'], mean, std)
        
        x_t = ME.TensorField(
            features=features,
            coordinates=normalized_coordinates,
            quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
            minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
            device=self.device,
        )

        # Clean up cache after tensor creation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return x_t

    def forward(self, x_full, x_full_sparse, x_part, t, pcd_part, mean, std):
        part_feat = self.partial_enc(x_part)
        
        if torch.all(pcd_part == 0):
            concatenated_features = torch.cat([part_feat.F, part_feat.F], dim=1)
        else:
            img, is_seen, point_loc_in_img = self.img_extractor(pcd_part.cuda())
            
            # Get the device of the input image
            device = img.device
            
            # Process the image with ResNet-50
            try:
                features = self.resnet_model(img)
                #print(f"ResNet encoding successful, features device: {features.device}")
            except Exception as e:
                print(f"ERROR in ResNet encoding: {str(e)}")
                # Emergency fallback
                print("Using fallback random features due to ResNet failure")
                B = img.shape[0]
                features = torch.randn(B, 2048, device=img.device)  # ResNet-50 outputs 2048 features
            
            # Reshape features to match the expected format
            B, C = features.shape  # [B, 2048]
            img_feat = features.reshape(B, 1, 1, C).permute(0, 3, 1, 2)
            img_feat = img_feat.reshape(-1, 10, 1, 2048)  # Adjusted for ResNet-50's 2048 features
            
            img_condition_emb, is_seen, point_loc = vanilla_upprojection(
                    img_feat, is_seen, point_loc_in_img, img_size=(224, 224),
                    n_points=self.hparams['data']['num_points'] // self.hparams['data'].get('upsample_ratio', 10),
                    vweights=None
                )
                
            img_feat = self.points_to_tensor_with_features(pcd_part, img_condition_emb, mean, std)
            img_feat = self.partial_enc_img(img_feat)
            
            # Concatenate feature matrices
            concatenated_features = torch.cat([part_feat.F, img_feat.F], dim=1)

        # Create a new SparseTensor with concatenated features
        part_feat = ME.SparseTensor(
            features=concatenated_features,
            coordinates=part_feat.C  # Keep original coordinates
        )
        
        out = self.model(x_full, x_full_sparse, part_feat, t)
        self.cleanup_cache()
        return out.reshape(t.shape[0],-1,3)

    def training_step(self, batch:dict, batch_idx):
        # Get optimizer and scheduler
        opt = self.optimizers()
        sch = self.lr_schedulers()
        
        # Clear cache at the beginning of each step
        self.cleanup_cache()
        
        # Use mixed precision training
        with torch.cuda.amp.autocast():
            # initial random noise
            noise = torch.randn(batch['pcd_full'].shape, device=self.device)
            
            # sample step t
            t = torch.randint(0, self.t_steps, size=(batch['pcd_full'].shape[0],)).cuda()
            t_sample = batch['pcd_full'] + self.q_sample(torch.zeros_like(batch['pcd_full']), t, noise)
            
            # replace the original points with the noise sampled
            x_full = self.points_to_tensor(t_sample, batch['mean'], batch['std'])

            # for classifier-free guidance switch between conditional and unconditional training
            if torch.rand(1) > self.hparams['train']['uncond_prob'] or batch['pcd_full'].shape[0] == 1:
                x_part = self.points_to_tensor(batch['pcd_part'], batch['mean'], batch['std'])
            else:
                x_part = self.points_to_tensor(
                    torch.zeros_like(batch['pcd_part']), torch.zeros_like(batch['mean']), torch.zeros_like(batch['std'])
                )

            denoise_t = self.forward(x_full, x_full.sparse(), x_part, t, batch['pcd_part'], batch['mean'], batch['std'])
            loss_mse = self.p_losses(denoise_t, noise)
            loss_mean = (denoise_t.mean())**2
            loss_std = (denoise_t.std() - 1.)**2
            loss = loss_mse + self.hparams['diff']['reg_weight'] * (loss_mean + loss_std)

        # Manual optimization with mixed precision
        opt.zero_grad()
        self.manual_backward(loss)
        
        # Manual gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        
        opt.step()
        
        # Update learning rate scheduler
        if sch is not None:
            sch.step()

        # Calculate metrics for logging
        with torch.no_grad():
            std_noise = (denoise_t - noise)**2
            
        # Log metrics
        self.log('train/loss_mse', loss_mse, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/loss_mean', loss_mean, on_step=True, on_epoch=True)
        self.log('train/loss_std', loss_std, on_step=True, on_epoch=True)
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/var', std_noise.var(), on_step=True, on_epoch=True)
        self.log('train/std', std_noise.std(), on_step=True, on_epoch=True)
        
        # Clean up cache at the end of training step
        self.cleanup_cache()

        return loss

    def validation_step(self, batch:dict, batch_idx):
        if batch_idx != 0:
            return

        self.model.eval()
        self.partial_enc.eval()
        self.partial_enc_img.eval()
        self.resnet_model.eval()
        
        # Clear cache before validation
        self.cleanup_cache()
        
        with torch.no_grad():
            gt_pts = batch['pcd_full'].detach().cpu().numpy()

            # for inference we get the partial pcd and sample the noise around the partial
            x_init = batch['pcd_part'].repeat(1, self.hparams['data'].get('upsample_ratio', 10), 1)
            x_feats = x_init + torch.randn(x_init.shape, device=self.device)
            x_full = self.points_to_tensor(x_feats, batch['mean'], batch['std'])
            x_part = self.points_to_tensor(batch['pcd_part'], batch['mean'], batch['std'])
            x_uncond = self.points_to_tensor(
                torch.zeros_like(batch['pcd_part']), torch.zeros_like(batch['mean']), torch.zeros_like(batch['std'])
            )

            x_gen_eval = self.p_sample_loop(x_init, x_full, x_part, x_uncond, gt_pts, batch['pcd_part'], batch['mean'], batch['std'])
            x_gen_eval = x_gen_eval.F.reshape((gt_pts.shape[0],-1,3))

            for i in range(len(batch['pcd_full'])):
                pcd_pred = SimplePointCloud()
                c_pred = x_gen_eval[i].cpu().detach().numpy()
                pcd_pred.set_points(c_pred)

                pcd_gt = SimplePointCloud()
                g_pred = batch['pcd_full'][i].cpu().detach().numpy()
                pcd_gt.set_points(g_pred)

                self.chamfer_distance.update(pcd_gt, pcd_pred)
                self.precision_recall.update(pcd_gt, pcd_pred)

        cd_mean, cd_std = self.chamfer_distance.compute()
        pr, re, f1 = self.precision_recall.compute_auc()

        self.log('val/cd_mean', cd_mean, on_step=True)
        self.log('val/cd_std', cd_std, on_step=True)
        self.log('val/precision', pr, on_step=True)
        self.log('val/recall', re, on_step=True)
        self.log('val/fscore', f1, on_step=True)
        
        # Clean up cache after validation
        self.cleanup_cache()

        return {'val/cd_mean': cd_mean, 'val/cd_std': cd_std, 'val/precision': pr, 'val/recall': re, 'val/fscore': f1}

    def valid_paths(self, filenames):
        output_paths = []
        skip = []

        for fname in filenames:
            seq_dir =  f'{self.logger.log_dir}/generated_pcd/{fname.split("/")[-3]}'
            ply_name = f'{fname.split("/")[-1].split(".")[0]}.ply'

            skip.append(path.isfile(f'{seq_dir}/{ply_name}'))
            makedirs(seq_dir, exist_ok=True)
            output_paths.append(f'{seq_dir}/{ply_name}')

        return np.all(skip), output_paths

    def test_step(self, batch:dict, batch_idx):
        self.model.eval()
        self.partial_enc.eval()
        self.partial_enc_img.eval()
        self.resnet_model.eval()
        
        # Clear cache before test
        self.cleanup_cache()
        
        with torch.no_grad():
            skip, output_paths = self.valid_paths(batch['filename'])

            if skip:
                print(f'Skipping generation from {output_paths[0]} to {output_paths[-1]}') 
                return {'test/cd_mean': 0., 'test/cd_std': 0., 'test/precision': 0., 'test/recall': 0., 'test/fscore': 0.}

            gt_pts = batch['pcd_full'].detach().cpu().numpy()

            x_init = batch['pcd_part'].repeat(1, self.hparams['data'].get('upsample_ratio', 10), 1)
            x_feats = x_init + torch.randn(x_init.shape, device=self.device)
            x_full = self.points_to_tensor(x_feats, batch['mean'], batch['std'])
            x_part = self.points_to_tensor(batch['pcd_part'], batch['mean'], batch['std'])
            x_uncond = self.points_to_tensor(
                torch.zeros_like(batch['pcd_part']), torch.zeros_like(batch['mean']), torch.zeros_like(batch['std'])
            )

            x_gen_eval = self.p_sample_loop(x_init, x_full, x_part, x_uncond, gt_pts, batch['pcd_part'], batch['mean'], batch['std'])
            x_gen_eval = x_gen_eval.F.reshape((gt_pts.shape[0],-1,3))

            for i in range(len(batch['pcd_full'])):
                pcd_pred = SimplePointCloud()
                c_pred = x_gen_eval[i].cpu().detach().numpy()
                dist_pts = np.sqrt(np.sum((c_pred)**2, axis=-1))
                dist_idx = dist_pts < self.hparams['data']['max_range']
                points = c_pred[dist_idx]
                max_z = x_init[i][...,2].max().item()
                min_z = (x_init[i][...,2].mean() - 2 * x_init[i][...,2].std()).item()
                pcd_pred.set_points(points[(points[:,2] < max_z) & (points[:,2] > min_z)])
                pcd_pred.set_colors(np.ones((len(points), 3)) * [1.0, 0.,0.])

                pcd_gt = SimplePointCloud()
                g_pred = batch['pcd_full'][i].cpu().detach().numpy()
                pcd_gt.set_points(g_pred)
                pcd_gt.set_colors(np.ones((len(g_pred), 3)) * [0., 1.,0.])
                
                print(f'Saving {output_paths[i]}')
                pcd_pred.save_ply(f'{output_paths[i]}')

                self.chamfer_distance.update(pcd_gt, pcd_pred)
                self.precision_recall.update(pcd_gt, pcd_pred)

        cd_mean, cd_std = self.chamfer_distance.compute()
        pr, re, f1 = self.precision_recall.compute_auc()
        print(f'CD Mean: {cd_mean}\tCD Std: {cd_std}')
        print(f'Precision: {pr}\tRecall: {re}\tF-Score: {f1}')

        self.log('test/cd_mean', cd_mean, on_step=True)
        self.log('test/cd_std', cd_std, on_step=True)
        self.log('test/precision', pr, on_step=True)
        self.log('test/recall', re, on_step=True)
        self.log('test/fscore', f1, on_step=True)
        
        # Clean up cache after test
        self.cleanup_cache()

        return {'test/cd_mean': cd_mean, 'test/cd_std': cd_std, 'test/precision': pr, 'test/recall': re, 'test/fscore': f1}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams['train']['lr'], betas=(0.9, 0.999))
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.5)
        scheduler = {
            'scheduler': scheduler, # lr * 0.5
            'interval': 'epoch', # interval is epoch-wise
            'frequency': 5, # after 5 epochs
        }

        return [optimizer], [scheduler]
