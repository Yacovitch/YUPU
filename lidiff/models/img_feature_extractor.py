import os
import sys

import torch
import warnings

from lidiff.models.prejection import RealisticProjection
import lidiff.models.clip as clip
import yaml

import numpy as np
from scipy.ndimage import zoom

warnings.filterwarnings("ignore")

PC_NUM = 1024

TRANS = -1.5

def load_projection_params(config_path=None):
    default = {'maxpoolz': 3,
               'maxpoolxy': 3,
               'maxpoolpadz': 1,
               'maxpoolpadxy': 1,
               'convz': 3,
               'convxy': 3,
               'convsigmaxy': 0.5,
               'convsigmaz': 1,
               'convpadz': 1,
               'convpadxy': 1,
               'imgbias': 0.,
               'depth_bias': 0.3,
               'obj_ratio': 0.95,
               'bg_clr': 0.0,
               'resolution': 224,
               'depth': 112}
    if config_path is None:
        return default
    try:
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        return cfg.get('projection', default)
    except Exception:
        return default
net = 'vit_b16'

cat2id = {'airplane': 0, 'bag': 1, 'cap': 2, 'car': 3, 'chair': 4,
          'earphone': 5, 'guitar': 6, 'knife': 7, 'lamp': 8, 'laptop': 9,
          'motorbike': 10, 'mug': 11, 'pistol': 12, 'rocket': 13, 'skateboard': 14, 'table': 15}


class Extractor(torch.nn.Module):
    def __init__(self, model, device, config_path=None):
        super(Extractor, self).__init__()

        self.model = model.encode_image
        self.device = device
        self.pc_views = None  # lazily initialized based on conditional PC size
        self.params_dict = load_projection_params(config_path)

    def mv_proj(self, pc):
        # Lazily create or refresh projection with current point count
        n_points = pc.shape[1]
        if (self.pc_views is None) or (self.pc_views.grid2image.n_points != n_points):
            self.pc_views = RealisticProjection(self.params_dict, self.device, n_points)

        img, is_seen, point_loc_in_img = self.pc_views.get_img(pc)
        img = img[:, :, 20:204, 20:204]
        point_loc_in_img = torch.ceil((point_loc_in_img - 20) * 224. / 184.)
        img = torch.nn.functional.interpolate(img, size=(224, 224), mode='bilinear', align_corners=True)
        return img, is_seen, point_loc_in_img

    def forward(self, pc, is_save=False):
        img, is_seen, point_loc_in_img = self.mv_proj(pc)
        print(img.shape)
        _, x = self.model(img)
        x = x / x.norm(dim=-1, keepdim=True)
        B, L, C = x.shape
        feat = x.reshape(B, 14, 14, C).permute(0, 3, 1, 2)
        # print(B, L, C, x.shape, is_seen.shape, point_loc_in_img.shape)
        #feat, is_seen, point_loc = vanilla_upprojection(feat, is_seen, point_loc, img_size=self.params_dict['resolution'],
                                                        #n_points=2048, vweights=None)
        return is_seen, point_loc_in_img, feat

    
    
class Extractor_img(torch.nn.Module):
    def __init__(self, device, n_points, config_path=None, params_override=None):
        super(Extractor_img, self).__init__()
        self.device = device
        params_dict = params_override if params_override is not None else load_projection_params(config_path)
        self.pc_views = RealisticProjection(params_dict, self.device, n_points= n_points)
        self.get_img = self.pc_views.get_img
        self.params_dict = params_dict

    def mv_proj(self, pc):
        img, is_seen, point_loc_in_img = self.get_img(pc)
        img = img[:, :, 20:204, 20:204]
        point_loc_in_img = torch.ceil((point_loc_in_img - 20) * 224. / 184.)
        img = torch.nn.functional.interpolate(img, size=(224, 224), mode='bilinear', align_corners=True)
        return img, is_seen, point_loc_in_img

    def forward(self, pc, is_save=False):
        img, is_seen, point_loc_in_img = self.mv_proj(pc.to(self.device))
        return img, is_seen, point_loc_in_img
    
