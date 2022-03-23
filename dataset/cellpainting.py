import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Normalize
from gulpio.transforms import ComposeVideo, RandHorFlipVideo, RandVerFlipVideo, RandomCropVideo, CenterCrop
from .utils import ResizeTensor, CropPatch

import numpy as np
import pandas as pd

import os

class CustomTransform(object):
    def __init__(self, mode, img_size=512, original_size=512):
        if mode == 'train':
            img_transforms = []
            video_transforms = [RandomCropVideo(original_size), RandHorFlipVideo(), RandVerFlipVideo()]
        elif mode == 'val' or mode == 'test':
            img_transforms = [CenterCrop(original_size),]
            video_transforms = []
        else:
            raise KeyError("mode %s is not valid, must be 'train' or 'val' or 'test'" % mode)

        self.transforms = ComposeVideo(img_transforms=img_transforms, video_transforms=video_transforms)
        self.to_tensor = ToTensor()
        self.normalize = Normalize((0.5, 0.5, 0.5, 0.5, 0.5), (1., 1., 1., 1., 1.))
        self.resize = ResizeTensor(image_size=img_size, original_size=original_size)
    
    def __call__(self, imgs):
        imgs = self.transforms(imgs)
        imgs = [self.to_tensor(img) for img in imgs]
        imgs = torch.cat(imgs, 0)
        imgs = self.normalize(imgs)
        imgs = self.resize(imgs)
        return imgs

class CellPaintingDataset(Dataset):
    ''' Base Dataset class '''

    def __init__(self, datadir, metafile, mode="train", img_size=512, featfile=None):
        self.datadir = datadir
        self.metadata = pd.read_csv(metafile)
        self.molfeats = pd.read_csv(featfile, index_col=1) if featfile is not None else None
        self.transforms = CustomTransform(mode=mode, img_size=img_size)

    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        try:
            sample = self.get_sample_img(idx)
            sample.update(self.get_sample_mol(idx))

        except Exception as e:
            print(e)
            return None
        
        return sample['image'], sample['feat']

    def load_img(self, key):
        ''' Load image from key '''
        img = np.load(os.path.join(self.datadir, "%s.npz" % key))
        img = img["sample"] # Shape 520 x 696 x 5
        img = [img[:,:,j] for j in range(5)]
        img = self.transforms(img)

        return img
    
    def get_sample_img(self, idx):
        '''Returns a dict corresponding to sample img for the provided index'''
        sample = self.metadata.iloc[idx]
        key = sample['SAMPLE_KEY']

        # load 5-channel image
        img = self.load_img(key)

        return {'key_img': key, 'image': img}

    def get_sample_mol(self, idx):
        ''' Returns a dict corresponding to sample molecule for the provided index'''
        sample = self.metadata.iloc[idx]
        smiles = sample['SMILES']

        if self.molfeats is not None:
            feat = self.molfeats.loc[smiles]['FEAT']
            feat = eval(f"np.array({feat})")
            feat = torch.from_numpy(feat).float()
            return {'key_chem': sample['SAMPLE_KEY'], 'feat': feat}

        return {'key_chem': sample['SAMPLE_KEY'], 'feat': smiles}

class CustomTransformPatch(CustomTransform):
    def __init__(self, mode, img_size=512, original_size=512, patch_size=64):
        super().__init__(mode, img_size=img_size, original_size=original_size)
        if mode == 'train':
            self.patch_crop = CropPatch(patch_size=patch_size, random=True)
        else:
            self.patch_crop = CropPatch(patch_size=patch_size, random=False)

    def __call__(self, imgs):
        imgs = super().__call__(imgs)
        imgs = self.patch_crop(imgs)
        return imgs

class CellPaintingPatchDataset(CellPaintingDataset):
    def __init__(self, datadir, metafile, mode="train", img_size=512):
        super().__init__(datadir=datadir, metafile=metafile, mode=mode, img_size=img_size)
        self.transforms = CustomTransformPatch(mode=mode, img_size=img_size)
