import torch
from torchvision.utils import save_image

import numpy as np
import random

from dataset import setup_dataloaders
from models.proglow import build_proglow_model
from utils import setup_logger, parse_json, save_img_as_npz

import os
import argparse
import sys

def setup_args():

    options = argparse.ArgumentParser()

    options.add_argument('--save-dir', action="store", default='results')
    options.add_argument('--datadir', action="store", default="data/images/")
    options.add_argument('--pt-config-file', action="store", default='config.json')

    options.add_argument('--train-metafile', action="store", default="data/metadata/datasplit_gen_train.csv")
    options.add_argument('--val-metafile', action="store", default="data/metadata/datasplit_gen_test_easy.csv")
    options.add_argument('--dataset', action="store", default="cell-painting")
    
    options.add_argument('--featfile', action="store", default=None)
    options.add_argument('--img-size', action="store", default=64, type=int)

    options.add_argument('--n_sample', default=30, type=int, help='number of samples')
    options.add_argument('--seed', action="store", default=42, type=int)
    options.add_argument('--batch-size', action="store", dest="batch_size", default=20, type=int)
    options.add_argument('--num-workers', action="store", dest="num_workers", default=0, type=int)

    # gpu options
    options.add_argument('--use-gpu', action="store_false", default=True)

    # debugging mode
    options.add_argument('--debug-mode', action="store_true", default=False)

    return options.parse_args()

def gen_test_images(args, logger):
   
    # setup model
    pt_config = parse_json(args.pt_config_file)
    z_shapes = pt_config['z_shapes']
    temp = pt_config['temp']

    net = build_proglow_model(pt_config['modules'])
    net.eval()

    logger.info(net)

    if args.use_gpu:
        net.cuda()

    # setup test dataloader
    args.use_nce_loss = False
    _, testloader = setup_dataloaders(args)

    # setup metafiles
    with open(os.path.join(args.save_dir, "metadata_real.csv"), 'a') as metafile:
        print(f"SMILES,SAMPLE_KEY", file=metafile)

    with open(os.path.join(args.save_dir, "metadata_gen.csv"), 'a') as metafile:
        print(f"SMILES,SAMPLE_KEY", file=metafile)
            
   
    # iterate through testloader
    for batch_idx, (real_sample, cond) in enumerate(testloader):

        batch_size = real_sample.size(0)

        with torch.no_grad(): # generate fake images for every condition in the test set

            # generate noise vector
            z_sample = []
            for z,t in zip(z_shapes, temp):
                z_new = torch.randn(batch_size, *z) * t
                z_sample.append(z_new.cuda() if args.use_gpu else z_new)

            # generate conditional sample
            gen_sample = net.reverse(z_sample, cond).cpu().data
        
        # save gen png
        for ch in range(5):
            ch_img = gen_sample[:,ch:ch+1,:,:]
            save_image(ch_img, os.path.join(args.save_dir, 'examples/%s_%s_gen.png' % (batch_idx, ch)), 
                       normalize=True, nrow=10, range=(-0.5, 0.5))
        
        # save real png
        for ch in range(5):
            ch_img = real_sample[:,ch:ch+1,:,:]
            save_image(ch_img, os.path.join(args.save_dir, 'examples/%s_%s_real.png' % (batch_idx, ch)), 
                       normalize=True, nrow=10, range=(-0.5, 0.5))
            
        # save each real, gen pair of samples as npz files individually
        for sample_idx, (real_img, gen_img, smiles) in enumerate(zip(real_sample, gen_sample, cond)):
            real_fname = os.path.join(args.save_dir, f"images/{batch_idx}_{sample_idx}_real")
            gen_fname = os.path.join(args.save_dir, f"images/{batch_idx}_{sample_idx}_gen")

            save_img_as_npz(real_img, real_fname)
            with open(os.path.join(args.save_dir, "metadata_real.csv"), 'a') as metafile:
                print(f"{smiles},{os.path.basename(real_fname)}", file=metafile)
            logger.debug("Saved img at %s" % real_fname)

            save_img_as_npz(gen_img, gen_fname)
            with open(os.path.join(args.save_dir, "metadata_gen.csv"), 'a') as metafile:
                print(f"{smiles},{os.path.basename(gen_fname)}", file=metafile)
            logger.debug("Saved img at %s" % gen_fname)
                

if __name__ == "__main__":
    args = setup_args()
    os.makedirs(os.path.join(args.save_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, "examples"), exist_ok=True)
    logger = setup_logger(name="gen", save_dir=args.save_dir)
    logger.info(" ".join(sys.argv))
    logger.info(args)
    gen_test_images(args, logger)
