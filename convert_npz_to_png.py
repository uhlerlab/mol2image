import numpy as np
from PIL import Image

import sys
import os

def load_img(key):
    ''' Load image from key '''
    img = np.load("%s.npz" % key)
    img = img["sample"]
    img = [img[:,:,j] for j in range(5)]

    return img

def convert_npz_to_png(data_dir, save_dir):
    # obtain file list
    fname_list = os.listdir(data_dir)

    for fname in fname_list:
        fname = os.path.splitext(fname)[0] # get basename without extension
        npz_img = load_img(os.path.join(data_dir, fname))
        
        for ch_idx, ch in enumerate(npz_img): # individually save 5 channels of image
            ch = Image.fromarray((ch*256).astype('uint8'))
            ch.save(os.path.join(save_dir, '%s_%s.png' % (fname, ch_idx)))

if __name__ == "__main__":
    data_dir = sys.argv[1]
    save_dir = sys.argv[2]
    os.makedirs(save_dir, exist_ok=False)
    convert_npz_to_png(data_dir, save_dir)
