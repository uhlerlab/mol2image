import torch
import torch.nn as nn

from chemprop.models.mpn import MPN

from types import SimpleNamespace

def get_default_args():
    '''Returns Namespace of arguments for MPN and mol2graph compat'''

    args = SimpleNamespace()
    args.hidden_size=512
    args.bias=True
    args.depth=10
    args.dropout=0.0
    args.activation='ReLU'
    args.undirected=False
    args.ffn_hidden_size=None
    args.atom_messages=False
    args.use_input_features=False
    args.features_only=False
    args.no_cache=False
    args.cuda = True

    return args

def get_MPN(depth=5, hidden_size=512):
    args = get_default_args()
    args.depth = depth
    args.hidden_size = hidden_size
    return MPN(args)
