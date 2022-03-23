import torch
from torch import optim
import logging
import os
import json
import numpy as np

def save_checkpoint(current_state, filename):
    torch.save(current_state, filename)

def load_checkpoint(model, checkpoint_file, key='state_dict'):
    model.load_state_dict(torch.load(checkpoint_file)[key])

def setup_optimizer(name, param_list):
    if name == 'sgd':
        return optim.SGD(param_list, momentum=0.9)
    elif name == 'adam':
        return optim.Adam(param_list)
    else:
        raise KeyError("%s is not a valid optimizer (must be one of ['sgd', adam']" % name)

def setup_lr_scheduler(name, optimizer):
    if name == 'none':
        return None
    elif name == 'step':
        return optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=10, gamma=0.5)
    
def setup_logger(name, save_dir):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(os.path.join(save_dir, '%s_info.log' % name))
    fh.setLevel(logging.INFO)
    
    fh_debug = logging.FileHandler(os.path.join(save_dir, '%s_debug.log' % name))
    fh_debug.setLevel(logging.DEBUG)

    logger.addHandler(fh)
    logger.addHandler(fh_debug)

    return logger

def parse_json(json_file):
    with open(json_file) as json_file:
        data = json.load(json_file)
    return data
    
def generate_config_file(args, pt_config, z_shapes, model_type, temp):
    modules = {'type': model_type, 'n_flow': args.n_flow, 
                'n_block': args.n_block, 'in_channel': args.n_channel, 
                'img_size': args.img_size, 'type': model_type,
                'checkpoint': None}

    if 'cond' in model_type:
        modules['n_cond'] = args.n_cond

    modules = [modules]

    if pt_config:
        modules += pt_config['modules']

    config = {'modules': modules, 'z_shapes': z_shapes, 'temp': temp}
    with open(os.path.join(args.save_dir, 'config.json'), 'w') as json_file:
        json.dump(config, json_file)
                
def save_img_as_npz(img, fname):
    img = img.numpy()
    img = img.transpose(1,2,0)
    img = img + 0.5 # undo normalization
    img_dict = {}
    img_dict['sample'] = img

    np.savez(fname, **img_dict)
