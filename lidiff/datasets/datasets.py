import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
from lidiff.datasets.dataloader.SemanticKITTITemporal import TemporalKITTISet
from lidiff.datasets.dataloader.SemanticKITTIGridTemporal import TemporalKITTIGridSet
from lidiff.datasets.dataloader.SemanticKITTITemporalNormal import TemporalKITTISetNormal
from lidiff.datasets.dataloader.SensatUrbanTemporal import TemporalSensat
from lidiff.datasets.dataloader.SensatUrbanTemporalNormal import TemporalSensatNormal
from lidiff.datasets.dataloader.YUPUTemporal import TemporalYUPUSet
from lidiff.datasets.dataloader.YUPUGridTemporal import TemporalYUPUGridSet
from lidiff.datasets.dataloader.YUPUTemporalNormal import TemporalYUPUNormalSet
from lidiff.utils.collations import SparseSegmentCollation, SparseSegmentCollationNormal
import warnings

warnings.filterwarnings('ignore')

__all__ = ['TemporalKittiDataModule']

class TemporalKittiDataModule(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def prepare_data(self):
        # Augmentations
        pass

    def setup(self, stage=None):
        # Create datasets
        pass

    def train_dataloader(self):
        collate = SparseSegmentCollation()

        data_set = TemporalKITTISet(
            data_dir=self.cfg['data']['data_dir'],
            seqs=self.cfg['data']['train'],
            split=self.cfg['data']['split'],
            resolution=self.cfg['data']['resolution'],
            num_points=self.cfg['data']['num_points'],
            upsample_ratio=self.cfg['data'].get('upsample_ratio', 10),
            max_range=self.cfg['data']['max_range'],
            dataset_norm=self.cfg['data']['dataset_norm'],
            std_axis_norm=self.cfg['data']['std_axis_norm'])
        loader = DataLoader(data_set, batch_size=self.cfg['train']['batch_size'], shuffle=True,
                            num_workers=self.cfg['train']['num_workers'], collate_fn=collate)
        return loader

    def val_dataloader(self, pre_training=True):
        collate = SparseSegmentCollation()

        data_set = TemporalKITTISet(
            data_dir=self.cfg['data']['data_dir'],
            seqs=self.cfg['data']['validation'],
            split='validation',
            resolution=self.cfg['data']['resolution'],
            num_points=self.cfg['data']['num_points'],
            upsample_ratio=self.cfg['data'].get('upsample_ratio', 10),
            max_range=self.cfg['data']['max_range'],
            dataset_norm=self.cfg['data']['dataset_norm'],
            std_axis_norm=self.cfg['data']['std_axis_norm'])
        loader = DataLoader(data_set, batch_size=1,#self.cfg['train']['batch_size'],
                            num_workers=self.cfg['train']['num_workers'], collate_fn=collate)
        return loader

    def test_dataloader(self):
        collate = SparseSegmentCollation()

        data_set = TemporalKITTISet(
            data_dir=self.cfg['data']['data_dir'],
            seqs=self.cfg['data']['validation'],
            split='validation',
            resolution=self.cfg['data']['resolution'],
            num_points=self.cfg['data']['num_points'],
            upsample_ratio=self.cfg['data'].get('upsample_ratio', 10),
            max_range=self.cfg['data']['max_range'],
            dataset_norm=self.cfg['data']['dataset_norm'],
            std_axis_norm=self.cfg['data']['std_axis_norm'])
        loader = DataLoader(data_set, batch_size=self.cfg['train']['batch_size'],
                             num_workers=self.cfg['train']['num_workers'], collate_fn=collate)
        return loader
    
class TemporalKittiNormalDataModule(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def prepare_data(self):
        # Augmentations
        pass

    def setup(self, stage=None):
        # Create datasets
        pass

    def train_dataloader(self):
        collate = SparseSegmentCollationNormal()

        data_set = TemporalKITTISetNormal(
            data_dir=self.cfg['data']['data_dir'],
            seqs=self.cfg['data']['train'],
            split=self.cfg['data']['split'],
            resolution=self.cfg['data']['resolution'],
            num_points=self.cfg['data']['num_points'],
            upsample_ratio=self.cfg['data'].get('upsample_ratio', 10),
            max_range=self.cfg['data']['max_range'],
            dataset_norm=self.cfg['data']['dataset_norm'],
            std_axis_norm=self.cfg['data']['std_axis_norm'])
        loader = DataLoader(data_set, batch_size=self.cfg['train']['batch_size'], shuffle=True,
                            num_workers=self.cfg['train']['num_workers'], collate_fn=collate)
        return loader

    def val_dataloader(self, pre_training=True):
        collate = SparseSegmentCollationNormal()

        data_set = TemporalKITTISetNormal(
            data_dir=self.cfg['data']['data_dir'],
            seqs=self.cfg['data']['validation'],
            split='validation',
            resolution=self.cfg['data']['resolution'],
            num_points=self.cfg['data']['num_points'],
            upsample_ratio=self.cfg['data'].get('upsample_ratio', 10),
            max_range=self.cfg['data']['max_range'],
            dataset_norm=self.cfg['data']['dataset_norm'],
            std_axis_norm=self.cfg['data']['std_axis_norm'])
        loader = DataLoader(data_set, batch_size=1,#self.cfg['train']['batch_size'],
                            num_workers=self.cfg['train']['num_workers'], collate_fn=collate)
        return loader

    def test_dataloader(self):
        collate = SparseSegmentCollationNormal()

        data_set = TemporalKITTISetNormal(
            data_dir=self.cfg['data']['data_dir'],
            seqs=self.cfg['data']['validation'],
            split='validation',
            resolution=self.cfg['data']['resolution'],
            num_points=self.cfg['data']['num_points'],
            upsample_ratio=self.cfg['data'].get('upsample_ratio', 10),
            max_range=self.cfg['data']['max_range'],
            dataset_norm=self.cfg['data']['dataset_norm'])
        loader = DataLoader(data_set, batch_size=self.cfg['train']['batch_size'],
                             num_workers=self.cfg['train']['num_workers'], collate_fn=collate)
        return loader
    
class TemporalSensatDataModule(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def prepare_data(self):
        # Augmentations
        pass

    def setup(self, stage=None):
        # Create datasets
        pass

    def train_dataloader(self):
        collate = SparseSegmentCollation()

        data_set = TemporalSensat(
            data_dir=self.cfg['data']['data_dir'],
            seqs=self.cfg['data']['train'],
            split=self.cfg['data']['split'],
            resolution=self.cfg['data']['resolution'],
            num_points=self.cfg['data']['num_points'],
            upsample_ratio=self.cfg['data'].get('upsample_ratio', 10),
            max_range=self.cfg['data']['max_range'],
            dataset_norm=self.cfg['data']['dataset_norm'],
            std_axis_norm=self.cfg['data']['std_axis_norm'])
        loader = DataLoader(data_set, batch_size=self.cfg['train']['batch_size'], shuffle=True,
                            num_workers=self.cfg['train']['num_workers'], collate_fn=collate)
        return loader

    def val_dataloader(self, pre_training=True):
        collate = SparseSegmentCollation()

        data_set = TemporalSensat(
            data_dir=self.cfg['data']['data_dir'],
            seqs=self.cfg['data']['validation'],
            split='validation',
            resolution=self.cfg['data']['resolution'],
            num_points=self.cfg['data']['num_points'],
            upsample_ratio=self.cfg['data'].get('upsample_ratio', 10),
            max_range=self.cfg['data']['max_range'],
            dataset_norm=self.cfg['data']['dataset_norm'],
            std_axis_norm=self.cfg['data']['std_axis_norm'])
        loader = DataLoader(data_set, batch_size=1,#self.cfg['train']['batch_size'],
                            num_workers=self.cfg['train']['num_workers'], collate_fn=collate)
        return loader

    def test_dataloader(self):
        collate = SparseSegmentCollation()

        data_set = TemporalSensat(
            data_dir=self.cfg['data']['data_dir'],
            seqs=self.cfg['data']['validation'],
            split='validation',
            resolution=self.cfg['data']['resolution'],
            num_points=self.cfg['data']['num_points'],
            upsample_ratio=self.cfg['data'].get('upsample_ratio', 10),
            max_range=self.cfg['data']['max_range'],
            dataset_norm=self.cfg['data']['dataset_norm'],
            std_axis_norm=self.cfg['data']['std_axis_norm'])
        loader = DataLoader(data_set, batch_size=self.cfg['train']['batch_size'],
                             num_workers=self.cfg['train']['num_workers'], collate_fn=collate)
        return loader
    
class TemporalKittiGridDataModule(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def prepare_data(self):
        # Augmentations
        pass

    def setup(self, stage=None):
        # Create datasets
        pass

    def train_dataloader(self):
        collate = SparseSegmentCollation()

        data_set = TemporalKITTIGridSet(
            data_dir=self.cfg['data']['data_dir'],
            seqs=self.cfg['data']['train'],
            split=self.cfg['data']['split'],
            resolution=self.cfg['data']['resolution'],
            num_points=self.cfg['data']['num_points'],
            upsample_ratio=self.cfg['data'].get('upsample_ratio', 10),
            max_range=self.cfg['data']['max_range'],
            dataset_norm=self.cfg['data']['dataset_norm'],
            std_axis_norm=self.cfg['data']['std_axis_norm'])
        loader = DataLoader(data_set, batch_size=self.cfg['train']['batch_size'], shuffle=True,
                            num_workers=self.cfg['train']['num_workers'], collate_fn=collate)
        return loader

    def val_dataloader(self, pre_training=True):
        collate = SparseSegmentCollation()

        data_set = TemporalKITTIGridSet(
            data_dir=self.cfg['data']['data_dir'],
            seqs=self.cfg['data']['validation'],
            split='validation',
            resolution=self.cfg['data']['resolution'],
            num_points=self.cfg['data']['num_points'],
            upsample_ratio=self.cfg['data'].get('upsample_ratio', 10),
            max_range=self.cfg['data']['max_range'],
            dataset_norm=self.cfg['data']['dataset_norm'],
            std_axis_norm=self.cfg['data']['std_axis_norm'])
        loader = DataLoader(data_set, batch_size=1,#self.cfg['train']['batch_size'],
                            num_workers=self.cfg['train']['num_workers'], collate_fn=collate)
        return loader

    def test_dataloader(self):
        collate = SparseSegmentCollation()

        data_set = TemporalKITTIGridSet(
            data_dir=self.cfg['data']['data_dir'],
            seqs=self.cfg['data']['validation'],
            split='validation',
            resolution=self.cfg['data']['resolution'],
            num_points=self.cfg['data']['num_points'],
            upsample_ratio=self.cfg['data'].get('upsample_ratio', 10),
            max_range=self.cfg['data']['max_range'],
            dataset_norm=self.cfg['data']['dataset_norm'],
            std_axis_norm=self.cfg['data']['std_axis_norm'])
        loader = DataLoader(data_set, batch_size=self.cfg['train']['batch_size'],
                             num_workers=self.cfg['train']['num_workers'], collate_fn=collate)
        return loader
        
class TemporalSensatNormalDataModule(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def train_dataloader(self):
        collate = SparseSegmentCollationNormal()

        data_set = TemporalSensatNormal(
            data_dir=self.cfg['data']['data_dir'],
            seqs=self.cfg['data']['train'],
            split=self.cfg['data']['split'],
            resolution=self.cfg['data']['resolution'],
            num_points=self.cfg['data']['num_points'],
            upsample_ratio=self.cfg['data'].get('upsample_ratio', 10),
            max_range=self.cfg['data']['max_range'],
        )
        loader = DataLoader(data_set, batch_size=self.cfg['train']['batch_size'], shuffle=True,
                            num_workers=self.cfg['train']['num_workers'], collate_fn=collate)
        return loader
    
    def val_dataloader(self, pre_training=True):
        collate = SparseSegmentCollationNormal()
        
        data_set = TemporalSensatNormal(
            data_dir=self.cfg['data']['data_dir'],
            seqs=self.cfg['data']['validation'],
            split='validation',
            resolution=self.cfg['data']['resolution'],
            num_points=self.cfg['data']['num_points'],
            upsample_ratio=self.cfg['data'].get('upsample_ratio', 10),
            max_range=self.cfg['data']['max_range'],
        )
        loader = DataLoader(data_set, batch_size=1,
                            num_workers=self.cfg['train']['num_workers'], collate_fn=collate)
        return loader

    def test_dataloader(self):
        collate = SparseSegmentCollationNormal()

        data_set = TemporalSensatNormal(
            data_dir=self.cfg['data']['data_dir'],
            seqs=self.cfg['data']['validation'],
            split='validation',
            resolution=self.cfg['data']['resolution'],
            num_points=self.cfg['data']['num_points'],
            upsample_ratio=self.cfg['data'].get('upsample_ratio', 10),
            max_range=self.cfg['data']['max_range'],
        )
        loader = DataLoader(data_set, batch_size=self.cfg['train']['batch_size'],
                             num_workers=self.cfg['train']['num_workers'], collate_fn=collate)
        return loader

class TemporalYUPUDataModule(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def train_dataloader(self):
        collate = SparseSegmentCollation()

        data_set = TemporalYUPUSet(
            data_dir=self.cfg['data']['data_dir'],
            seqs=self.cfg['data']['train'],
            split=self.cfg['data']['split'],
            resolution=self.cfg['data']['resolution'],
            num_points=self.cfg['data']['num_points'],
            upsample_ratio=self.cfg['data'].get('upsample_ratio', 10),
            max_range=self.cfg['data']['max_range'],
        )
        loader = DataLoader(data_set, batch_size=self.cfg['train']['batch_size'], shuffle=True,
                            num_workers=self.cfg['train']['num_workers'], collate_fn=collate)
        return loader

    def val_dataloader(self, pre_training=True):
        collate = SparseSegmentCollation()

        data_set = TemporalYUPUSet(
            data_dir=self.cfg['data']['data_dir'],
            seqs=self.cfg['data']['validation'],
            split='validation',
            resolution=self.cfg['data']['resolution'],
            num_points=self.cfg['data']['num_points'],
            upsample_ratio=self.cfg['data'].get('upsample_ratio', 10),
            max_range=self.cfg['data']['max_range'],
        )
        loader = DataLoader(data_set, batch_size=1,
                            num_workers=self.cfg['train']['num_workers'], collate_fn=collate)
        return loader

    def test_dataloader(self):
        collate = SparseSegmentCollation()

        data_set = TemporalYUPUSet(
            data_dir=self.cfg['data']['data_dir'],
            seqs=self.cfg['data']['validation'],
            split='validation',
            resolution=self.cfg['data']['resolution'],
            num_points=self.cfg['data']['num_points'],
            upsample_ratio=self.cfg['data'].get('upsample_ratio', 10),
            max_range=self.cfg['data']['max_range'],
        )
        loader = DataLoader(data_set, batch_size=self.cfg['train']['batch_size'],
                             num_workers=self.cfg['train']['num_workers'], collate_fn=collate)
        return loader

class TemporalYUPUGridDataModule(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def train_dataloader(self):
        collate = SparseSegmentCollation()

        data_set = TemporalYUPUGridSet(
            data_dir=self.cfg['data']['data_dir'],
            seqs=self.cfg['data']['train'],
            split=self.cfg['data']['split'],
            resolution=self.cfg['data']['resolution'],
            num_points=self.cfg['data']['num_points'],
            upsample_ratio=self.cfg['data'].get('upsample_ratio', 10),
            max_range=self.cfg['data']['max_range'],
        )
        loader = DataLoader(data_set, batch_size=self.cfg['train']['batch_size'], shuffle=True,
                            num_workers=self.cfg['train']['num_workers'], collate_fn=collate)
        return loader

    def val_dataloader(self, pre_training=True):
        collate = SparseSegmentCollation()

        data_set = TemporalYUPUGridSet(
            data_dir=self.cfg['data']['data_dir'],
            seqs=self.cfg['data']['validation'],
            split='validation',
            resolution=self.cfg['data']['resolution'],
            num_points=self.cfg['data']['num_points'],
            upsample_ratio=self.cfg['data'].get('upsample_ratio', 10),
            max_range=self.cfg['data']['max_range'],
        )
        loader = DataLoader(data_set, batch_size=1,
                            num_workers=self.cfg['train']['num_workers'], collate_fn=collate)
        return loader

    def test_dataloader(self):
        collate = SparseSegmentCollation()

        data_set = TemporalYUPUGridSet(
            data_dir=self.cfg['data']['data_dir'],
            seqs=self.cfg['data']['validation'],
            split='validation',
            resolution=self.cfg['data']['resolution'],
            num_points=self.cfg['data']['num_points'],
            upsample_ratio=self.cfg['data'].get('upsample_ratio', 10),
            max_range=self.cfg['data']['max_range'],
        )
        loader = DataLoader(data_set, batch_size=self.cfg['train']['batch_size'],
                             num_workers=self.cfg['train']['num_workers'], collate_fn=collate)
        return loader

class TemporalYUPUNormalDataModule(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def train_dataloader(self):
        collate = SparseSegmentCollationNormal()

        data_set = TemporalYUPUNormalSet(
            data_dir=self.cfg['data']['data_dir'],
            seqs=self.cfg['data']['train'],
            split=self.cfg['data']['split'],
            resolution=self.cfg['data']['resolution'],
            num_points=self.cfg['data']['num_points'],
            max_range=self.cfg['data']['max_range'],
            upsample_ratio=self.cfg['data'].get('upsample_ratio', 4),
            synthetic_downsample=self.cfg['data'].get('synthetic_downsample_train', False),
            synthetic_downsample_method=self.cfg['data'].get('synthetic_downsample_method', 'random'),
            synthetic_normal_k=self.cfg['data'].get('synthetic_normal_k', 30),
        )
        loader = DataLoader(data_set, batch_size=self.cfg['train']['batch_size'], shuffle=True,
                            num_workers=self.cfg['train']['num_workers'], collate_fn=collate)
        return loader

    def val_dataloader(self, pre_training=True):
        collate = SparseSegmentCollationNormal()

        data_set = TemporalYUPUNormalSet(
            data_dir=self.cfg['data']['data_dir'],
            seqs=self.cfg['data']['validation'],
            split='validation',
            resolution=self.cfg['data']['resolution'],
            num_points=self.cfg['data']['num_points'],
            max_range=self.cfg['data']['max_range'],
            upsample_ratio=self.cfg['data'].get('upsample_ratio', 4),
            synthetic_downsample=self.cfg['data'].get('synthetic_downsample_train', False),
            synthetic_downsample_method=self.cfg['data'].get('synthetic_downsample_method', 'random'),
            synthetic_normal_k=self.cfg['data'].get('synthetic_normal_k', 30),
        )
        loader = DataLoader(data_set, batch_size=1,
                            num_workers=self.cfg['train']['num_workers'], collate_fn=collate)
        return loader

    def test_dataloader(self):
        collate = SparseSegmentCollationNormal()

        data_set = TemporalYUPUNormalSet(
            data_dir=self.cfg['data']['data_dir'],
            seqs=self.cfg['data']['test'],
            split='test',
            resolution=self.cfg['data']['resolution'],
            num_points=self.cfg['data']['num_points'],
            max_range=self.cfg['data']['max_range'],
            upsample_ratio=self.cfg['data'].get('upsample_ratio', 4),
            synthetic_downsample=self.cfg['data'].get('synthetic_downsample_train', False),
            synthetic_downsample_method=self.cfg['data'].get('synthetic_downsample_method', 'random'),
            synthetic_normal_k=self.cfg['data'].get('synthetic_normal_k', 30),
        )
        loader = DataLoader(data_set, batch_size=self.cfg['train']['batch_size'],
                             num_workers=self.cfg['train']['num_workers'], collate_fn=collate)
        return loader

dataloaders = {
    'KITTI': TemporalKittiDataModule,
    'KITTIGrid': TemporalKittiGridDataModule,
    'KITTINormal': TemporalKittiNormalDataModule,
    'Sensat': TemporalSensatDataModule,
    'SensatNormal': TemporalSensatNormalDataModule,
    'YUPU': TemporalYUPUDataModule,
    'YUPUGrid': TemporalYUPUGridDataModule,
    'YUPUNormal': TemporalYUPUNormalDataModule,
}
