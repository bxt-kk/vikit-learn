#! /usr/bin/env python3
from typing import Dict, Any
import json

from torch.utils.data import DataLoader
from torch.optim import AdamW, Adam, SGD, NAdam
import torch

from vklearn.trainer.trainer import Trainer
from vklearn.trainer.tasks import Classification as Task
from vklearn.models.trimnetclf import TrimNetClf as Model
from vklearn.datasets.images_folder import ImagesFolder
from vklearn.datasets.oxford_iiit_pet import OxfordIIITPet
from vklearn.datasets.places365 import Places365


def load_from_json(path:str) -> Dict[str, Any]:
    with open(path) as f:
        cfg = json.load(f)
    return cfg


def main(cfg:Dict[str, Any]):
    dataset_root = cfg['dataset']['root']

    train_transforms, test_transforms = Model.get_transforms(
        cfg['dataset']['transform'])

    dataset_type = cfg['dataset']['type']
    dataset_opts = dict()
    if dataset_type == 'ImagesFolder':
        Dataset = ImagesFolder
        dataset_opts['extensions'] = cfg['dataset']['extensions']
    elif dataset_type == 'OxfordIIITPet':
        Dataset = OxfordIIITPet
        dataset_opts['target_types'] = 'binary-category'
    elif dataset_type == 'Places365':
        Dataset = Places365

    train_data = Dataset(
        dataset_root,
        split='train',
        transforms=train_transforms,
        **dataset_opts)
    test_data = Dataset(
        dataset_root,
        split='val',
        transforms=test_transforms,
        **dataset_opts)

    batch_size = cfg['dataset']['batch_size']
    num_workers = cfg['dataset']['num_workers']

    train_loader = DataLoader(
        train_data, batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers)
    test_loader = DataLoader(
        test_data, batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=num_workers)

    device_name = cfg['task']['device']
    if device_name == 'auto':
        device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_name)
    model  = Model(categories=train_data.classes, **cfg['model'])
    task   = Task(
        model=model,
        device=device,
        metric_start_epoch=cfg['task']['metric_start_epoch'],
        fit_features_start=cfg['task']['fit_features_start'],
    )

    optim_method = {
        'AdamW': AdamW,
        'Adam': Adam,
        'SGD': SGD,
        'NAdam': NAdam,
    }[cfg['trainer']['optim_method']]

    trainer = Trainer(
        task,
        output=cfg['trainer']['output'],
        train_loader=train_loader,
        test_loader=test_loader,
        checkpoint=cfg['trainer']['checkpoint'],
        drop_optim=cfg['trainer']['drop_optim'],
        drop_lr_scheduler=cfg['trainer']['drop_lr_scheduler'],
        optim_method=optim_method,
        lr=cfg['trainer']['lr'],
        weight_decay=cfg['trainer']['weight_decay'],
        lrf=cfg['trainer']['lrf'],
        T_num=cfg['trainer']['T_num'],
        grad_steps=cfg['trainer']['grad_steps'],
        epochs=cfg['trainer']['epochs'],
        show_step=cfg['trainer']['show_step'],
        save_epoch=cfg['trainer']['save_epoch'],
    )

    trainer.initialize()
    trainer.fit()


def entry():
    import argparse

    parser = argparse.ArgumentParser(
        description='Image classification trainer')
    parser.add_argument(
        'path',
        metavar='configure path',
        type=str,
        help='The parameters configure of the trainer',
    )

    args = parser.parse_args()
    cfg = load_from_json(args.path)
    main(cfg)


if __name__ == "__main__":
    entry()
