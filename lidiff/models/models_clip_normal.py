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

from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import LightningDataModule
from lidiff.utils.collations import *
from lidiff.utils.metrics import ChamferDistance, PrecisionRecall
from diffusers import DPMSolverMultistepScheduler

import sys

#clip model
from lidiff.models.img_feature_extractor import Extractor, Extractor_img
from lidiff.models.unprojection import vanilla_upprojection
import lidiff.models.clip as clip

import matplotlib.pyplot as plt
#import cv2

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

class CrossModalMixer(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, dropout=0.0, num_modalities=3,
                 fuse='concat', out_dim=512):
        super().__init__()
        assert fuse in ['concat', 'mean', 'max', 'gate', 'cls']
        self.embed_dim = embed_dim
        self.num_modalities = num_modalities
        self.fuse = fuse
        self.out_dim = out_dim

        # attention over modality tokens
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

        # optional learned gates for fusion
        if fuse == 'gate':
            self.gate_proj = nn.Linear(embed_dim, 1)

        # optional CLS token for fusion
        if fuse == 'cls':
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            nn.init.trunc_normal_(self.cls_token, std=0.02)

        # output projection
        fused_in = {
            'concat': embed_dim * num_modalities,
            'mean':   embed_dim,
            'max':    embed_dim,
            'gate':   embed_dim,
            'cls':    embed_dim,
        }[fuse]
        self.out_proj = nn.Sequential(
            nn.Linear(fused_in, out_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: ME.SparseTensor) -> ME.SparseTensor:
        # Expect x.F shape [N, num_modalities*embed_dim]
        feats = x.F
        assert feats.shape[1] == self.num_modalities * self.embed_dim, \
            f"Expected features dim {self.num_modalities*self.embed_dim}, got {feats.shape[1]}"

        # split into modality tokens and stack -> [N, M, D]
        tokens = feats.view(feats.size(0), self.num_modalities, self.embed_dim)

        # (optional) add a CLS token for fusion
        if self.fuse == 'cls':
            cls = self.cls_token.expand(tokens.size(0), -1, -1)  # [N,1,D]
            tokens = torch.cat([cls, tokens], dim=1)             # [N,1+M,D]

        # Transformer block over modality dimension
        h = self.norm1(tokens)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        tokens = tokens + attn_out

        h = self.norm2(tokens)
        ffn_out = self.ffn(h)
        tokens = tokens + ffn_out  # [N, (1+M), D] or [N, M, D]

        # fuse across modality tokens
        if self.fuse == 'concat':
            fused = tokens.reshape(tokens.size(0), -1)  # [N, M*D]
        elif self.fuse == 'mean':
            fused = tokens.mean(dim=1)                  # [N, D]
        elif self.fuse == 'max':
            fused, _ = tokens.max(dim=1)                # [N, D]
        elif self.fuse == 'gate':
            # learned soft gates per modality
            gates = torch.softmax(self.gate_proj(tokens).squeeze(-1), dim=1)  # [N, M]
            fused = (tokens * gates.unsqueeze(-1)).sum(dim=1)                 # [N, D]
        elif self.fuse == 'cls':
            fused = tokens[:, 0, :]                      # [N, D] (CLS)

        fused = self.out_proj(fused)  # [N, out_dim] (e.g., 512)

        return ME.SparseTensor(
            features=fused,
            coordinate_map_key=x.coordinate_map_key,
            coordinate_manager=x.coordinate_manager
        )

class DiffusionPoints(LightningModule):
    def __init__(self, hparams:dict, data_module: LightningDataModule = None, config_path=None):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.data_module = data_module
        self.config_path = config_path

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
        self.partial_enc_img = minknet.MinkowskiEncoder256(in_channels=512, feature_dim=256)
        self.partial_enc_normal = minknet.MinkGlobalEnc(in_channels=3, out_channels=self.hparams['model']['out_dim'])
        
        #Clip Model
        model_name = 'ViT-B/32'
        
        # Make sure we're using CUDA for CLIP model
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"FORCING CLIP to use CUDA device: {device}")
            # Ensure CUDA is initialized before loading CLIP
            _ = torch.zeros(1, device=device)  # Force CUDA initialization
        else:
            device = torch.device('cpu')
            print(f"WARNING: CUDA not available, using CPU for CLIP: {device}")
        
        # Use try-except for robust model loading
        try:
            # Directly load to CUDA 
            print(f"Loading CLIP model to device: {device}")
            clip_model, _ = clip.load(model_name, device=device)
            print(f"Successfully loaded CLIP model to {device}")
        except Exception as e:
            print(f"ERROR loading CLIP model: {str(e)}")
            # One more fallback attempt
            print("Trying CPU loading as fallback...")
            try:
                clip_model, _ = clip.load(model_name, device="cpu")
                clip_model = clip_model.to(device)
                print(f"Successfully moved CLIP model to {device} after CPU loading")
            except Exception as e2:
                print(f"CRITICAL: All CLIP loading attempts failed: {str(e2)}")
                raise RuntimeError("Failed to load CLIP model")
        
        # Extra step to ensure ALL parameters are on CUDA
        clip_model_device = next(clip_model.parameters()).device
        print(f"CLIP model parameter device before fixing: {clip_model_device}")
        
        if torch.cuda.is_available() and clip_model_device.type != 'cuda':
            print("⚠️ CLIP model is not on CUDA despite loading attempt - forcefully moving ALL parameters")
            for param in clip_model.parameters():
                param.data = param.data.to(device)
            print(f"CLIP model parameter device after fixing: {next(clip_model.parameters()).device}")
        
        # Set up parallel processing if multiple GPUs are available
        if torch.cuda.device_count() > 1:
            print(f"Setting up DataParallel with {torch.cuda.device_count()} GPUs")
            clip_model = torch.nn.DataParallel(clip_model, device_ids=list(range(torch.cuda.device_count())))

        # Store the full CLIP model
        # Freeze CLIP parameters and set to eval to reduce memory/compute; still DataParallel-capable
        for p in clip_model.parameters():
            p.requires_grad = False
        clip_model.eval()
        
        self.clip_model = clip_model
        
        proj_params = None
        try:
            proj_params = self.hparams.get('projection', None)
        except Exception:
            proj_params = None
            
        self.img_extractor = Extractor_img(device = device, n_points = self.hparams['data']['num_points']//self.hparams['data']['upsample_ratio'], params_override=proj_params)
        
        self.modality_mixer = CrossModalMixer(embed_dim=256, num_heads=8, dropout=0.0, num_modalities=3, fuse='mean', out_dim=512)

        self.model = minknet.MinkUNetDiffClip(in_channels=3, out_channels=self.hparams['model']['out_dim'])

        self.chamfer_distance = ChamferDistance()
        self.precision_recall = PrecisionRecall(self.hparams['data']['resolution'],2*self.hparams['data']['resolution'],100)

        self.w_uncond = self.hparams['train']['uncond_w']

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

    def classfree_forward(self, x_t, x_cond, x_uncond, normals, t, pcd_part, mean, std):
        x_t_sparse = x_t.sparse()
        x_cond = self.forward(x_t, x_t_sparse, x_cond, normals, t, pcd_part, mean, std)            
        x_uncond = self.forward(x_t, x_t_sparse, x_uncond, normals, t, torch.zeros_like(pcd_part), torch.zeros_like(mean), torch.zeros_like(std))

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

    def reset_partial_pcd(self, x_part, x_uncond, x_mean, x_std, normals):
        # Recreate TensorFields to avoid coordinate map stacking and align coordinate maps across modalities
        new_x_part = self.points_to_tensor(x_part.F.reshape(x_mean.shape[0], -1, 3).detach(), x_mean, x_std)
        new_x_uncond = self.points_to_tensor(
            torch.zeros_like(new_x_part.F.reshape(x_mean.shape[0], -1, 3)), torch.zeros_like(x_mean), torch.zeros_like(x_std)
        )
        # Recover per-point normal features per batch and rebuild with the SAME coordinate map as new_x_part
        normals_feats = normals.F.reshape(x_mean.shape[0], -1, normals.F.shape[1]).detach()
        new_normals = self.points_to_tensor_with_features(
            new_x_part.F.reshape(x_mean.shape[0], -1, 3).detach(), normals_feats, x_mean, x_std, like=new_x_part
        )

        return new_x_part, new_x_uncond, new_normals

    def p_sample_loop(self, x_init, x_t, x_cond, x_uncond, gt_pts, x_part_batch, x_mean, x_std, normals):
        pcd = SimplePointCloud()
        self.scheduler_to_cuda()

        for t in tqdm(range(len(self.dpm_scheduler.timesteps))):
            t = torch.ones(gt_pts.shape[0]).cuda().long() * self.dpm_scheduler.timesteps[t].cuda()
            
            noise_t = self.classfree_forward(x_t, x_cond, x_uncond, normals, t, x_part_batch, x_mean, x_std)
            input_noise = x_t.F.reshape(t.shape[0],-1,3) - x_init
            x_t = x_init + self.dpm_scheduler.step(noise_t, t[0], input_noise)['prev_sample']
            x_t = self.points_to_tensor(x_t, x_mean, x_std)

            # this is needed otherwise minkEngine will keep "stacking" coords maps over the x_part and x_uncond
            # i.e. memory leak
            x_cond, x_uncond, normals = self.reset_partial_pcd(x_cond, x_uncond, x_mean, x_std, normals)
            torch.cuda.empty_cache()

        makedirs(f'{self.logger.log_dir}/generated_pcd/', exist_ok=True)

        return x_t

    def p_losses(self, y, noise):
        return F.mse_loss(y, noise)

    def forward(self, x_full, x_full_sparse, x_part, normals, t, pcd_part, mean, std):
        #print('forward')
        #print('x_part: ', x_part.shape)
        part_feat = self.partial_enc(x_part)
        normal_feat = self.partial_enc_normal(normals)
        
        if torch.all(pcd_part == 0):
            concatenated_features = torch.cat([part_feat.F, part_feat.F, part_feat.F], dim=1)
        else:
            img, is_seen, point_loc_in_img = self.img_extractor(pcd_part.cuda())

            # Get the device of the input image
            device = img.device
            #print(f"Image device in forward: {device}")
            
            # CRITICAL: Make sure CLIP is on GPU and stays there
            if torch.cuda.is_available():
                # Force CLIP to GPU - extra check
                target_device = torch.device('cuda')
                current_device = next(self.clip_model.parameters()).device
                
                if current_device.type != 'cuda':
                    print(f"⚠️ CRITICAL: CLIP model is on {current_device}, forcefully moving to {target_device}")
                    # Manual parameter transfer to ensure everything moves
                    for param in self.clip_model.parameters():
                        param.data = param.data.to(target_device)
                    self.clip_model = self.clip_model.to(target_device)
                
                # Make sure the image is on GPU too
                if img.device.type != 'cuda':
                    print(f"Moving image from {img.device} to CUDA")
                    img = img.to(target_device)
            
            # Process the image with the CLIP model
            #print(f"CLIP model device before encoding: {next(self.clip_model.parameters()).device}")
            #print(f"Image device before encoding: {img.device}")
            
            try:
                if isinstance(self.clip_model, torch.nn.DataParallel):
                    _, x = self.clip_model.module.encode_image(img)
                else:
                    _, x = self.clip_model.encode_image(img)
                
                #print(f"CLIP encoding successful, features device: {x.device}")
            except Exception as e:
                print(f"ERROR in CLIP encoding: {str(e)}")
                # Emergency fallback
                print("Using fallback random features due to CLIP failure")
                B, _, _, _ = img.shape
                x = torch.randn(B, 1, 512, device=img.device)
            
            x = x / x.norm(dim=-1, keepdim=True)
            B, L, C = x.shape
            img_feat = x.reshape(B, 7, 7, C).permute(0, 3, 1, 2)
            img_feat = img_feat.reshape(-1, 10, 49, 512)
            
            img_condition_emb, is_seen, point_loc = vanilla_upprojection(
                    img_feat, is_seen, point_loc_in_img, img_size=(224, 224), n_points=self.hparams['data']['num_points']//self.hparams['data']['upsample_ratio'], vweights=None
                )
                
            img_feat = self.points_to_tensor_with_features(pcd_part, img_condition_emb, mean, std)
            img_feat = self.partial_enc_img(img_feat)
            # Align normal features to part feature coordinates to ensure equal row counts
            matched_normal_feats = self.model.match_part_to_full(part_feat, normal_feat)
            matched_img_feats = self.model.match_part_to_full(part_feat, img_feat)
            concatenated_features = torch.cat([part_feat.F, matched_img_feats, matched_normal_feats], dim=1)

        # Create a new SparseTensor with concatenated features
        part_feat = ME.SparseTensor(
            features=concatenated_features,
            coordinates=part_feat.C  # Keep original coordinates
        )
        
        part_feat = self.modality_mixer(part_feat)
        
        out = self.model(x_full, x_full_sparse, part_feat, t)
        torch.cuda.empty_cache()
        return out.reshape(t.shape[0],-1,3)
    
    
    
    def points_to_tensor(self, x_feats, mean, std):
        x_feats = ME.utils.batched_coordinates(list(x_feats[:]), dtype=torch.float32, device=self.device)
        #print('mean: ', mean, 'std: ', std)
        x_coord = x_feats.clone()
        x_coord[:,1:] = feats_to_coord(x_feats[:,1:], self.hparams['data']['resolution'], mean, std)
        #print('points_to_tensor')
        #print('x_feats: ', x_feats.shape)
        #print('x_coord: ', x_coord.shape)
        x_t = ME.TensorField(
            features=x_feats[:,1:],
            coordinates=x_coord,
            quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
            minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
            device=self.device,
        )

        torch.cuda.empty_cache()
        #print('x_t: ', x_t.shape)
        return x_t

    def points_to_tensor_with_features(self, x_feats, features, mean, std, like=None):
        """
        Converts 3D point coordinates and additional 512 features into an ME TensorField.

        Args:
            x_feats (torch.Tensor): Shape [1, num_points, 3], containing (x, y, z) coordinates.
            features (torch.Tensor): Shape [1, num_points, 512], containing 512 features per point.
            mean (torch.Tensor): Mean for coordinate normalization.
            std (torch.Tensor): Standard deviation for coordinate normalization.

        Returns:
            ME.TensorField: MinkowskiEngine TensorField with quantized coordinates and input feature dimension.
        """
        # Ensure features is a 2D matrix [total_points, feature_dim]
        # Accept inputs of shape [B, N, C] or [N, C]
        if features.dim() == 3:
            batch_size, num_points, feature_dim = features.shape
            features = features.reshape(batch_size * num_points, feature_dim)
        elif features.dim() == 2:
            # already [N, C]
            pass
        else:
            raise AssertionError(
                f"Expected features to be 2D or 3D, got order-{features.dim()} tensor."
            )

        # Convert to MinkowskiEngine batched coordinates (adds batch indices)
        coordinates = ME.utils.batched_coordinates(list(x_feats[:]), dtype=torch.float32, device=self.device)

        # Normalize coordinates
        normalized_coordinates = coordinates.clone()
        normalized_coordinates[:, 1:] = feats_to_coord(coordinates[:, 1:], self.hparams['data']['resolution'], mean, std)
        #print('points_to_tensor_with_features')
        #print('features: ', features.shape)
        #print('coordinates: ', normalized_coordinates.shape)
        # Create ME.TensorField with correct feature assignments
        # Align device/dtype and validate size match with coordinates
        features = features.to(device=self.device, dtype=torch.float32)
        assert features.shape[0] == coordinates.shape[0], \
            f"Mismatched rows: features {features.shape[0]} vs coordinates {coordinates.shape[0]}"

        x_t = ME.TensorField(
            features=features,  # [B*N, C]
            coordinates=normalized_coordinates,
            quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
            minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
            device=self.device,
        )

        torch.cuda.empty_cache()  # Free unused GPU memory
        #print('x_t: ', x_t.shape)
        return x_t
    
    def training_step(self, batch:dict, batch_idx):
        # initial random noise
        torch.cuda.empty_cache()
        noise = torch.randn(batch['pcd_full'].shape, device=self.device)
        
        # sample step t
        t = torch.randint(0, self.t_steps, size=(batch['pcd_full'].shape[0],)).cuda()
        t_sample = batch['pcd_full'] + self.q_sample(torch.zeros_like(batch['pcd_full']), t, noise)# directly add noise without normalize
        
        # sample q at step t
        # we sample noise towards zero to then add to each point the noise (without normalizing the pcd)
        
        
        '''
        t[:] = 999
        t_sample = batch['pcd_full'] + self.q_sample(torch.zeros_like(batch['pcd_full']), t, noise)# directly add noise without normalize
        
        pcd_diff = o3d.geometry.PointCloud()
        pcd_diff.points = o3d.utility.Vector3dVector(t_sample[0].cpu().detach().numpy())
        pcd_diff.estimate_normals()
        o3d.io.write_point_cloud(f'/nas2/jacob/LiDiff/lidiff/visualization/{999}.ply', pcd_diff)
        
        t[:] = 0
        t_sample = batch['pcd_full'] + self.q_sample(torch.zeros_like(batch['pcd_full']), t, noise)# directly add noise without normalize
        
        pcd_diff = o3d.geometry.PointCloud()
        pcd_diff.points = o3d.utility.Vector3dVector(t_sample[0].cpu().detach().numpy())
        pcd_diff.estimate_normals()
        o3d.io.write_point_cloud(f'/nas2/jacob/LiDiff/lidiff/visualization/{000}.ply', pcd_diff)
        
        
        t[:] = 250
        t_sample = batch['pcd_full'] + self.q_sample(torch.zeros_like(batch['pcd_full']), t, noise)# directly add noise without normalize
        
        pcd_diff = o3d.geometry.PointCloud()
        pcd_diff.points = o3d.utility.Vector3dVector(t_sample[0].cpu().detach().numpy())
        pcd_diff.estimate_normals()
        o3d.io.write_point_cloud(f'/nas2/jacob/LiDiff/lidiff/visualization/{250}.ply', pcd_diff)
        
        t[:] = 500
        t_sample = batch['pcd_full'] + self.q_sample(torch.zeros_like(batch['pcd_full']), t, noise)# directly add noise without normalize
        
        pcd_diff = o3d.geometry.PointCloud()
        pcd_diff.points = o3d.utility.Vector3dVector(t_sample[0].cpu().detach().numpy())
        pcd_diff.estimate_normals()
        o3d.io.write_point_cloud(f'/nas2/jacob/LiDiff/lidiff/visualization/{500}.ply', pcd_diff)
        
        t[:] = 750
        t_sample = batch['pcd_full'] + self.q_sample(torch.zeros_like(batch['pcd_full']), t, noise)# directly add noise without normalize
        
        pcd_diff = o3d.geometry.PointCloud()
        pcd_diff.points = o3d.utility.Vector3dVector(t_sample[0].cpu().detach().numpy())
        pcd_diff.estimate_normals()
        o3d.io.write_point_cloud(f'/nas2/jacob/LiDiff/lidiff/visualization/{750}.ply', pcd_diff)
        sys.exit("generation complete")
        '''
        
        
        # replace the original points with the noise sampled
        x_full = self.points_to_tensor(t_sample, batch['mean'], batch['std'])

        # for classifier-free guidance switch between conditional and unconditional training
        #print('pcd_part size:', batch['pcd_part'].shape)
        if torch.rand(1) > self.hparams['train']['uncond_prob'] or batch['pcd_full'].shape[0] == 1:
            x_part = self.points_to_tensor(batch['pcd_part'], batch['mean'], batch['std'])
        else:
            x_part = self.points_to_tensor(
                torch.zeros_like(batch['pcd_part']), torch.zeros_like(batch['mean']), torch.zeros_like(batch['std'])
            )
            
        x_normals = self.points_to_tensor_with_features(batch['pcd_part'], batch['normals'], batch['mean'], batch['std'], like=x_part)

        denoise_t = self.forward(x_full, x_full.sparse(), x_part, x_normals, t, batch['pcd_part'], batch['mean'], batch['std'])
        loss_mse = self.p_losses(denoise_t, noise)
        loss_mean = (denoise_t.mean())**2
        loss_std = (denoise_t.std() - 1.)**2
        loss = loss_mse + self.hparams['diff']['reg_weight'] * (loss_mean + loss_std)

        std_noise = (denoise_t - noise)**2
        self.log('train/loss_mse', loss_mse)
        self.log('train/loss_mean', loss_mean)
        self.log('train/loss_std', loss_std)
        self.log('train/loss', loss)
        self.log('train/var', std_noise.var())
        self.log('train/std', std_noise.std())
        torch.cuda.empty_cache()

        return loss

    def validation_step(self, batch:dict, batch_idx):
        if batch_idx != 0:
            return

        self.model.eval()
        self.partial_enc.eval()
        self.partial_enc_normal.eval()
        self.partial_enc_img.eval()
        self.clip_model.eval()
        self.modality_mixer.eval()
        
        with torch.no_grad():
            gt_pts = batch['pcd_full'].detach().cpu().numpy()

            # for inference we get the partial pcd and sample the noise around the partial
            x_init = batch['pcd_part'].repeat(1,10,1)
            x_feats = x_init + torch.randn(x_init.shape, device=self.device)
            x_full = self.points_to_tensor(x_feats, batch['mean'], batch['std'])
            x_part = self.points_to_tensor(batch['pcd_part'], batch['mean'], batch['std'])
            x_normals = self.points_to_tensor_with_features(batch['pcd_part'], batch['normals'], batch['mean'], batch['std'], like=x_part)
            x_uncond = self.points_to_tensor(
                torch.zeros_like(batch['pcd_part']), torch.zeros_like(batch['mean']), torch.zeros_like(batch['std'])
            )

            x_gen_eval = self.p_sample_loop(x_init, x_full, x_part, x_uncond, gt_pts, batch['pcd_part'], batch['mean'], batch['std'], x_normals)
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
        print(f'CD Mean: {cd_mean}\tCD Std: {cd_std}')
        print(f'Precision: {pr}\tRecall: {re}\tF-Score: {f1}')

        self.log('val/cd_mean', cd_mean, on_step=True)
        self.log('val/cd_std', cd_std, on_step=True)
        self.log('val/precision', pr, on_step=True)
        self.log('val/recall', re, on_step=True)
        self.log('val/fscore', f1, on_step=True)
        torch.cuda.empty_cache()

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
        self.clip_model.eval()
        self.modality_mixer.eval()
        with torch.no_grad():
            skip, output_paths = self.valid_paths(batch['filename'])

            if skip:
                print(f'Skipping generation from {output_paths[0]} to {output_paths[-1]}') 
                return {'test/cd_mean': 0., 'test/cd_std': 0., 'test/precision': 0., 'test/recall': 0., 'test/fscore': 0.}

            gt_pts = batch['pcd_full'].detach().cpu().numpy()

            x_init = batch['pcd_part'].repeat(1,10,1)
            x_feats = x_init + torch.randn(x_init.shape, device=self.device)
            x_full = self.points_to_tensor(x_feats, batch['mean'], batch['std'])
            x_part = self.points_to_tensor(batch['pcd_part'], batch['mean'], batch['std'])
            x_normals = self.points_to_tensor_with_features(batch['pcd_part'], batch['normals'], batch['mean'], batch['std'])
            x_uncond = self.points_to_tensor(
                torch.zeros_like(batch['pcd_part']), torch.zeros_like(batch['mean']), torch.zeros_like(batch['std'])
            )
            

            x_gen_eval = self.p_sample_loop(x_init, x_full, x_part, x_uncond, gt_pts, batch['pcd_part'], torch.zeros_like(batch['pcd_part']), batch['mean'], batch['std'], x_normals)
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
        torch.cuda.empty_cache()

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
        #return [optimizer]

#######################################
# Modules
#######################################


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

    # Normalize spatial coordinates (x, y) to image dimensions for BEV
    x = ((points[:, 0] + 1) / 2 * (width - 1)).long()  # Use x as x
    y = ((points[:, 2] + 1) / 2 * (height - 1)).long()  # Use z as y for top-down view

    # Normalize depth (height) to [0, depth_bins)
    z = ((points[:, 1] - points[:, 1].min()) / (points[:, 1].max() - points[:, 1].min()) * (depth_bins - 1)).long()

    # Clamp indices to valid ranges
    x = x.clamp(0, width - 1)
    y = y.clamp(0, height - 1)
    z = z.clamp(0, depth_bins - 1)

    # Initialize depth image
    depth_image = torch.zeros(height, width, dtype=torch.long, device=points.device)

    # Populate depth image (taking the nearest depth value per pixel)
    depth_image[y, x] = z

    return depth_image
