from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

from .cellpainting import CellPaintingDataset, CellPaintingPatchDataset

import numpy as np

def my_collate(batch):
    ''' Custom collate function that filters out empty samples '''
    batch = list(filter (lambda x:x is not None, batch))
    return default_collate(batch)

def data_sampler(dataset, batch_size, num_workers, shuffle=True):
    ''' Helper for sampling data from full dataset '''
    loader = DataLoader(dataset, shuffle=shuffle, drop_last=True, batch_size=batch_size, num_workers=num_workers, collate_fn=my_collate)
    loader = iter(loader)

    while True:
        try:
            yield next(loader)

        except StopIteration:
            loader = DataLoader(dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers, collate_fn=my_collate)
            loader = iter(loader)
            yield next(loader)

def setup_dataloaders(args):
    if args.dataset == 'cell-painting':
        trainset = CellPaintingDataset(datadir=args.datadir, metafile=args.train_metafile, img_size=args.img_size, featfile=args.featfile)
        valset = CellPaintingDataset(datadir=args.datadir, metafile=args.val_metafile, img_size=args.img_size, featfile=args.featfile)
    elif args.dataset == 'cell-painting-patch':
        assert(args.img_size > 64)
        print('loading patch dataset')
        trainset = CellPaintingPatchDataset(datadir=args.datadir, metafile=args.train_metafile, img_size=args.img_size)
        valset = CellPaintingPatchDataset(datadir=args.datadir, metafile=args.val_metafile, img_size=args.img_size)
    else:
        raise KeyError('Dataset %s is not valid' % args.dataset)

    trainloader = iter(data_sampler(dataset=trainset, batch_size=args.batch_size, num_workers=args.num_workers))

    if valset is not None:
        valloader = DataLoader(dataset=valset, batch_size=args.batch_size, num_workers=args.num_workers, 
                               drop_last=False, shuffle=args.use_nce_loss)
    else:
        valloader = None

    return trainloader, valloader
