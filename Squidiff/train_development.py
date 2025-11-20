#!/usr/bin/env python
"""
Squidiff Training Script for Development Dataset
- No drug structure (no SMILES, no drug_dose)
- No Group information
- Uses only gene expression and PCW as condition
- Official default settings: lr=1e-4, batch_size=64, hidden_sizes=2048, latent_dim=60
"""

import io
import os
import socket

import torch as th
import torch.distributed as dist
import argparse
from datetime import datetime
from Squidiff import dist_util, logger

from torch.utils.data import Dataset, DataLoader
import scanpy as sc
import numpy as np

from Squidiff.resample import create_named_schedule_sampler
from Squidiff.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from Squidiff.train_util import TrainLoop, plot_loss

GPUS_PER_NODE = 1


class DevelopmentDataset(Dataset):
    """Simple dataset for development data - no drug structure, dummy group for compatibility"""
    def __init__(self, adata):
        if type(adata.X) == np.ndarray:
            self.features = th.tensor(adata.X, dtype=th.float32)
        else:
            self.features = th.tensor(adata.X.toarray(), dtype=th.float32)

        # Use PCW as condition (convert categorical to float)
        self.pcw = th.tensor(adata.obs['pcw'].astype(float).values, dtype=th.float32)

        # Add dummy group for compatibility with train_util (not used in model)
        self.group = np.zeros(len(adata), dtype=np.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return {
            'feature': self.features[idx],
            'pcw': self.pcw[idx],
            'group': self.group[idx]  # Dummy group
        }


def prepared_data_development(data_dir=None, batch_size=64):  # Official default: batch_size=64
    """Prepare development data loader"""
    train_adata = sc.read_h5ad(data_dir)

    _data_dataset = DevelopmentDataset(train_adata)

    dataloader = DataLoader(
        _data_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    return dataloader


def run_training(args):
    logger.configure(dir=args['logger_path'])
    logger.log("*********creating model and diffusion**********")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args['schedule_sampler'], diffusion)

    logger.log("creating data loader...")
    data = prepared_data_development(
        data_dir=args['data_path'],
        batch_size=args['batch_size']
    )

    start_time = datetime.now()
    logger.log(f'**********training started at {start_time} **********')

    train_ = TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args['batch_size'],
        microbatch=args['microbatch'],
        lr=args['lr'],
        ema_rate=args['ema_rate'],
        log_interval=args['log_interval'],
        save_interval=args['save_interval'],
        resume_checkpoint=args['resume_checkpoint'],
        use_fp16=args['use_fp16'],
        fp16_scale_growth=args['fp16_scale_growth'],
        schedule_sampler=schedule_sampler,
        weight_decay=args['weight_decay'],
        lr_anneal_steps=args['lr_anneal_steps'],
        use_drug_structure=False,  # No drug structure
        comb_num=1,
    )
    train_.run_loop()

    end_time = datetime.now()
    during_time = (end_time - start_time).seconds / 60
    logger.log(f'start time: {start_time} end_time: {end_time} time:{during_time} min')

    return train_.loss_list


def parse_args():
    """Parse command-line arguments and update with default values."""
    default_args = {}
    default_args.update(model_and_diffusion_defaults())
    updated_args = {
        'data_path': '',
        'schedule_sampler': 'uniform',
        'lr': 1e-4,  # Official default
        'weight_decay': 0.0,  # Official default (no weight decay)
        'lr_anneal_steps': 100000,  # Official default (100k steps)
        'batch_size': 64,  # Official default
        'microbatch': -1,
        'ema_rate': '0.9999',
        'log_interval': 1e4,
        'save_interval': 1e4,
        'resume_checkpoint': '',
        'use_fp16': False,
        'fp16_scale_growth': 1e-3,
        'gene_size': 300,  # 300 genes in the dataset
        'output_dim': 300,
        'num_layers': 3,
        'class_cond': False,
        'use_encoder': True,
        'diffusion_steps': 1000,
        'logger_path': '',
        'use_drug_structure': False,  # No drug structure
        'comb_num': 1,
        'use_ddim': False,  # Use full DDPM, not DDIM
    }
    default_args.update(updated_args)

    parser = argparse.ArgumentParser(description='Squidiff Development Training')

    for key, value in default_args.items():
        if isinstance(value, bool):
            parser.add_argument(f'--{key}', default=value, type=lambda x: (str(x).lower() == 'true'), help=f'{key} (default: {value})')
        elif isinstance(value, float):
            parser.add_argument(f'--{key}', default=value, type=float, help=f'{key} (default: {value})')
        elif isinstance(value, int):
            parser.add_argument(f'--{key}', default=value, type=int, help=f'{key} (default: {value})')
        else:
            parser.add_argument(f'--{key}', default=value, type=type(value), help=f'{key} (default: {value})')

    args = parser.parse_args()
    updated_args = vars(args)

    if updated_args['logger_path'] == '':
        logger.log('ERROR:Please specify the logger path --logger_path.')
        raise ValueError("Logger path is required. Please specify the logger path.")

    if updated_args['data_path'] == '':
        logger.log("ERROR:Please specify the data path --data_path.")
        raise ValueError("Dataset path is required. Please specify the path where the training adata is.")

    return updated_args


if __name__ == "__main__":
    args_train = parse_args()
    print('**************training args*************')
    print(args_train)
    losses = run_training(args_train)

    plot_loss(losses, args_train)
