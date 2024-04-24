#! /usr/bin/env python3
from typing import Any, Dict
from pprint import pprint
from collections import defaultdict

import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from math import ceil


class Trainer:

    def __init__(
            self,
            model:        nn.Module,
            train_loader: DataLoader,
            output:       str,
            checkpoint:   str=None,
            valid_loader: DataLoader=None,
            test_loader:  DataLoader=None,
            lr:           float=1e-3,
            weight_decay: float=0.,
            epochs:       int=1,
            show_step:    int=100,
            save_epoch:   int=1,
            device:       torch.device | str=None,
            ):
        
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif isinstance(device, str):
            device = torch.device(device)
        self.device = device

        self.dataloaders = dict(
            train=train_loader,
            valid=valid_loader,
            test=test_loader,
        )

        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay

        self.checkpoint = checkpoint

        self.optimizer:torch.optim.Optimizer = None

        self.epochs = epochs

        self.show_step = show_step

        self.save_epoch = save_epoch

        self.output = output

    def setting_optimizer(self):
        return torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay)

    def initialize(self):
        print('Preparing ...')
        print('device:', self.device)
        for kind, loader in self.dataloaders.items():
            if loader is None: continue
            print(f'{kind} dataset:', loader.dataset)
            pprint(dict(
                batch_size=loader.batch_size,
                num_workers=loader.num_workers,
            ))
            
        self.model = self.model.to(self.device)
        self.optimizer = self.setting_optimizer()
        print('optimizer:', self.optimizer)

        if self.checkpoint is not None:
            state_dict = torch.load(self.checkpoint)
            print('checkpoint:', self.checkpoint)
            self.model.load_state_dict(state_dict['model'])
            self.optimizer.load_state_dict(state_dict['optim'])

    def train_on_step(
            self,
            epoch:  int,
            step:   int,
            sample: Any,
            logs:   Dict[str, Any],
        ):
        # raise ValueError('method `train_on_step` is uninitialized')

        inputs, target_index, target_labels, target_bboxes = sample
        inputs = inputs.to(self.device)
        target_labels = target_labels.to(self.device)
        target_bboxes = target_bboxes.to(self.device)

        model = self.model
        optimizer = self.optimizer
        dataloader = self.dataloaders['train']

        outputs = model(inputs)
        losses = model.calc_loss(outputs, target_index, target_labels, target_bboxes)
        loss = losses['loss']

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            scores = model.calc_score(outputs, target_index, target_labels, target_bboxes)

        for k, v in losses.items():
            logs['loss'][k] += v.item()
        for k, v in scores.items():
            logs['score'][k] += v.item()

        if (step + 1) % self.show_step == 0:
            print('epoch: {}/{}, step: {}/{}'.format(
                epoch + 1,
                self.epochs,
                step + 1,
                ceil(len(dataloader.dataset) / dataloader.batch_size),
            ))
            print('losses:', {k: round(v / (step + 1), 5) for k, v in logs['loss'].items()})
            print('scores:', {k: round(v / (step + 1), 5) for k, v in logs['score'].items()})

    def valid_on_step(
            self,
            epoch:  int,
            step:   int,
            sample: Any,
            logs:   Dict[str, Any],
        ):

        inputs, target_index, target_labels, target_bboxes = sample
        inputs = inputs.to(self.device)
        target_labels = target_labels.to(self.device)
        target_bboxes = target_bboxes.to(self.device)

        model = self.model
        dataloader = self.dataloaders['valid']

        with torch.no_grad():
            outputs = model(inputs)
            losses = model.calc_loss(outputs, target_index, target_labels, target_bboxes)
            scores = model.calc_score(outputs, target_index, target_labels, target_bboxes)

        for k, v in losses.items():
            logs['loss'][k] += v.item()
        for k, v in scores.items():
            logs['score'][k] += v.item()

        if (step + 1) % self.show_step == 0:
            print('epoch: {}/{}, step: {}/{}'.format(
                epoch + 1,
                self.epochs,
                step + 1,
                ceil(len(dataloader.dataset) / dataloader.batch_size),
            ))
            print('losses:', {k: round(v / (step + 1), 5) for k, v in logs['loss'].items()})
            print('scores:', {k: round(v / (step + 1), 5) for k, v in logs['score'].items()})

    def test_on_step(
            self,
            epoch:  int,
            step:   int,
            sample: Any,
            logs:   Dict[str, Any],
        ):

        inputs, target_index, target_labels, target_bboxes = sample
        inputs = inputs.to(self.device)
        target_labels = target_labels.to(self.device)
        target_bboxes = target_bboxes.to(self.device)

        model = self.model

        with torch.no_grad():
            outputs = model(inputs)
            losses = model.calc_loss(outputs, target_index, target_labels, target_bboxes)
            scores = model.calc_score(outputs, target_index, target_labels, target_bboxes)

        for k, v in losses.items():
            logs['loss'][k] += v.item()
        for k, v in scores.items():
            logs['score'][k] += v.item()

    def end_test_on_epoch(self, epoch:int, step:int, logs:Dict[str, Any]):
        dataloader = self.dataloaders['test']
        print('epoch: {}/{}, step: {}/{}'.format(
            epoch + 1,
            self.epochs,
            step + 1,
            ceil(len(dataloader.dataset) / dataloader.batch_size),
        ))
        print('losses:', {k: round(v / (step + 1), 5) for k, v in logs['loss'].items()})
        print('scores:', {k: round(v / (step + 1), 5) for k, v in logs['score'].items()})
        print('metrics:', {k: round(v / (step + 1), 5) for k, v in logs['metric'].items()})

    def fit(self):
        print('-' * 80)
        print('Training ...')
        model = self.model.train()
        optimizer = self.optimizer
        for epoch in range(self.epochs):
            logs = dict(
                loss=defaultdict(float),
                score=defaultdict(float),
            )
            train_loader = self.dataloaders['train']
            for step, sample in enumerate(train_loader):
                self.train_on_step(epoch, step, sample, logs)
                # self.valid_on_step(epoch, step, sample, logs)

            logs = dict(
                loss=defaultdict(float),
                score=defaultdict(float),
                metric=defaultdict(float),
            )
            model = model.eval()
            test_loader = self.dataloaders['test']
            if test_loader is not None:
                for step, sample in enumerate(test_loader):
                    self.test_on_step(epoch, step, sample, logs)
                self.end_test_on_epoch(epoch, step, logs)

            if (epoch + 1) % self.save_epoch == 0:
                checkpoint_filename = '{}-{}.pt'.format(self.output, epoch)
                print('save checkpoint -> {}'.format(checkpoint_filename))
                torch.save({
                    'model': model.state_dict(),
                    'anchors': model.anchors,
                    'cell_size': model.cell_size,
                    'optim': optimizer.state_dict()}, checkpoint_filename)

        checkpoint_filename = self.output + '.pt'
        print('finished and save checkpoint -> {}'.format(checkpoint_filename))
        torch.save({
            'model': model.state_dict(),
            'anchors': model.anchors,
            'cell_size': model.cell_size,
            'optim': optimizer.state_dict()}, checkpoint_filename)

        model_filename = self.output + '_model.pt'
        print('model -> {}'.format(model_filename))
        torch.save({
            'model': model.cpu().state_dict(),
            'anchors': model.anchors,
            'cell_size': model.cell_size}, model_filename)
