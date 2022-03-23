import torch
import torch.nn.functional as F
from torchvision import transforms

import numpy as np

class ResizeTensor(object):
    def __init__(self, image_size, original_size=256):
        self.image_size = image_size
        self.num_downscales = int(np.log2(original_size)) - int(np.log2(image_size))
        self.weight = torch.FloatTensor([[ 1/4, 1/4, 1/4, 1/4],]).view(1, 4, 1, 1, 1)

    def squeeze(self, input):
        n_channel, height, width = input.shape
        squeezed = input.view(n_channel, height // 2, 2, width // 2, 2)
        squeezed = squeezed.permute(2, 4, 0, 1, 3)
        squeezed = squeezed.contiguous().view(1, 4, n_channel, height // 2, width // 2)

        out = F.conv3d(squeezed, self.weight)
        return out.squeeze()

    def __call__(self, input):
        output = input
        for _ in range(self.num_downscales):
            output = self.squeeze(output)
        return output

class AddJitter(object):
    def __init__(self, n_bits=8):
        self.a = 1/(2**n_bits)

    def __call__(self, input):
        return input + self.a * torch.rand_like(input)

class CropPatch(object):
    def __init__(self, patch_size, random=True):
        self.patch_size = patch_size
        self.random = random

    def __call__(self, imgs):
        assert(len(imgs.shape) == 3)
        n_channel, width, height = imgs.size(0), imgs.size(1), imgs.size(2)

        if self.random:
            w = np.random.randint(0, width-self.patch_size)
            h = np.random.randint(0, height-self.patch_size)
        else:
            w = 0
            h = 0

        return imgs[:, w:w+self.patch_size, h:h+self.patch_size]
