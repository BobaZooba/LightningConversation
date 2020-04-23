import os
import numpy as np

from abc import ABC
from argparse import ArgumentParser

from src import data, neuro

import pytorch_lightning as pl
import torch
from torch import nn


class LightningConversation(pl.LightningModule, ABC):

    PAD_TOKEN = '<PAD>'
    SEP_TOKEN = '▁<SEP>'
    CONTEXT_TOKEN = '▁<CTX>'

    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams

        with open(os.path.join(self.hparams.data_dir, 'vocab.txt')) as file_object:
            self.vocab = file_object.read().split('\n')

        self.pad_index = self.vocab.index(self.PAD_TOKEN)
        self.sep_index = self.vocab.index(self.SEP_TOKEN)

        if self.CONTEXT_TOKEN in self.vocab:
            self.context_index = self.vocab.index(self.CONTEXT_TOKEN)
        else:
            self.context_index = None

        batching_strategy = data.BatchingStrategy(batching_type=self.hparams.batching_type,
                                                  batch_size=self.hparams.batch_size,
                                                  max_length=self.hparams.max_length)

        self.train_data = batching_strategy.load_data(data_dir=os.path.join(self.hparams.data_dir,
                                                                            'train'),
                                                      num_workers=self.hparams.num_workers)
        self.validation_data = batching_strategy.load_data(data_dir=os.path.join(self.hparams.data_dir,
                                                                                 'validation'),
                                                           num_workers=self.hparams.num_workers)

        self.batch_preparing = data.BatchPreparing(sep_index=self.sep_index,
                                                   context_index=self.context_index,
                                                   pad_index=self.pad_index)

        if self.hparams.criterion in ['label_smoothing', 'ls']:
            self.criterion = neuro.LabelSmoothingLoss(smoothing=self.hparams.smoothing,
                                                      use_kl=self.hparams.use_kl,
                                                      ignore_index=self.pad_index)
        elif self.hparams.criterion in ['unlikelihood', 'unlike']:
            # TODO добавить unlikelihood loss
            # https://arxiv.org/pdf/1908.04319.pdf
            # TODO добавить
            # https://www.aclweb.org/anthology/N18-1033.pdf
            raise ValueError('Not available now')
        else:
            self.criterion = nn.CrossEntropyLoss(ignore_index=self.pad_index)

    def configure_optimizers(self):
        """
        adam - faster convergence
        sgd with momentum - better convergence
        TODO add novograd
        """

        if self.hparams.optimizer == 'sgd':
            optimizer = torch.optim.SGD(params=self.parameters(),
                                        lr=self.hparams.learning_rate,
                                        weight_decay=self.hparams.weight_decay,
                                        momentum=self.hparams.momentum,
                                        nesterov=self.hparams.nesterov)
        else:
            optimizer = torch.optim.AdamW(params=self.parameters(),
                                          lr=self.hparams.learning_rate,
                                          weight_decay=self.hparams.weight_decay)

        # TODO другие шедуллеры
        if self.hparams.lr_scheduler == 'noam':
            scheduler = neuro.NoamScheduler(optimizer,
                                            model_dim=self.hparams.model_dim,
                                            warmup_steps=self.hparams.warmup_steps)
            return [optimizer], [scheduler]
        else:
            return optimizer

    def train_dataloader(self):
        train_dataset = data.ConversationDataset(data=self.train_data)

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=1,
            collate_fn=self.batch_preparing.collate
        )

        return train_loader

    def val_dataloader(self):
        validation_dataset = data.ConversationDataset(data=self.validation_data)

        validation_loader = torch.utils.data.DataLoader(
            validation_dataset,
            batch_size=1,
            collate_fn=self.batch_preparing.collate
        )

        return validation_loader

    @staticmethod
    def add_model_specific_args(parent_parser):

        parser = ArgumentParser(parents=[parent_parser])

        # model
        parser.add_argument('--model_dim', type=int, default=768)
        parser.add_argument('--num_heads', type=int, default=12)
        parser.add_argument('--feed_forward_dim', type=int, default=3072)
        parser.add_argument('--num_layers', type=int, default=12)
        parser.add_argument('--response_segment_index', type=int, default=1)
        parser.add_argument('--query_segment_index', type=int, default=2)
        parser.add_argument('--context_segment_index', type=int, default=3)
        parser.add_argument('--weight_tying', action='store_true')
        parser.add_argument('--vocab_size', type=int, default=32000)
        parser.add_argument('--n_positions', type=int, default=65)
        parser.add_argument('--dropout', type=float, default=0.1)
        parser.add_argument('--initializer_range', type=float, default=0.02)

        # loss
        parser.add_argument('--criterion', type=str, default='label_smoothing')
        parser.add_argument('--smoothing', type=float, default=0.1)
        parser.add_argument('--use_kl', action='store_true')

        # optimizers & schedulers
        parser.add_argument('--optimizer', type=str, default='adam')
        parser.add_argument('--learning_rate', type=float, default=0.001)
        parser.add_argument('--weight_decay', type=float, default=0.)
        parser.add_argument('--momentum', type=float, default=0.9)
        parser.add_argument('--nesterov', action='store_true')
        parser.add_argument('--warmup_steps', type=int, default=4000)
        parser.add_argument('--lr_scheduler', type=str, default='none')

        return parser


class LightningDialogGPT(LightningConversation):

    def __init__(self, hparams):
        super().__init__(hparams=hparams)

        self.model = neuro.GPT(
            model_dim=self.hparams.model_dim,
            num_heads=self.hparams.num_heads,
            feed_forward_dim=self.hparams.feed_forward_dim,
            num_layers=self.hparams.num_layers,
            query_segment_index=self.hparams.query_segment_index,
            context_segment_index=self.hparams.context_segment_index,
            weight_tying=self.hparams.weight_tying,
            vocab_size=self.hparams.vocab_size,
            n_positions=self.hparams.n_positions,
            dropout=self.hparams.dropout,
            padding_idx=self.pad_index,
            initializer_range=self.hparams.initializer_range
        )

    def forward(self, source_sequence, segment_indices, position_indices):

        logits = self.model(source_sequence, segment_indices, position_indices)

        return logits

    def training_step(self, batch, batch_idx):
        source_sequence, target_sequence, segment_indices, position_indices = batch

        logits = self.forward(source_sequence, segment_indices, position_indices)

        prediction, target = logits.reshape(-1, logits.size(-1)), target_sequence.view(-1)

        loss = self.criterion(prediction, target)

        log = {
            'train_loss': loss.item(),
            'train_perplexity_ls': np.exp(loss.item()),
            'batch_size': source_sequence.size(0),
            'length': source_sequence.size(1)
        }

        return {'loss': loss, 'log': log}

    def validation_step(self, batch, batch_idx):
        source_sequence, target_sequence, segment_indices, position_indices = batch

        logits = self.forward(source_sequence, segment_indices, position_indices)

        prediction = logits.reshape(-1, logits.size(-1)).detach().cpu()
        target = target_sequence.view(-1).detach().cpu()

        validation_loss = self.validation_criterion(prediction, target)
        validation_smoothed_loss = self.criterion(prediction, target)

        return {'val_loss': validation_loss, 'val_smoothed_loss': validation_smoothed_loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()

        log = {
            'val_loss': avg_loss,
            'val_perplexity': np.exp(avg_loss.item())
        }

        return {'val_loss': avg_loss, 'log': log}


class LightningDialoUnifiedTransformer(LightningConversation):

    def __init__(self, hparams):
        super().__init__(hparams)

        self.model = neuro.UnifiedTransformer(
            model_dim=self.hparams.model_dim,
            num_heads=self.hparams.num_heads,
            feed_forward_dim=self.hparams.feed_forward_dim,
            num_layers=self.hparams.num_layers,
            response_segment_index=self.hparams.response_segment_index,
            query_segment_index=self.hparams.query_segment_index,
            context_segment_index=self.hparams.context_segment_index,
            weight_tying=self.hparams.weight_tying,
            vocab_size=self.hparams.vocab_size,
            n_positions=self.hparams.n_positions,
            dropout=self.hparams.dropout,
            padding_idx=self.pad_index,
            initializer_range=self.hparams.initializer_range,
            masking_type=self.hparams.model_type
        )

        self.seq2seq_prob = self.hparams.seq2seq_prob

    def forward(self, source_sequence, segment_indices, position_indices, target_sequence):

        logits, targets = self.model(source_sequence, segment_indices, position_indices, target_sequence)

        return logits, targets

    def training_step(self, batch, batch_idx):
        source_sequence, target_sequence, segment_indices, position_indices = batch

        if self.seq2seq_prob > 0 and np.random.random() < self.seq2seq_prob:
            self.model.set_seq2seq()
        else:
            self.model.set_causal()

        logits, targets = self.forward(source_sequence, segment_indices, position_indices, target_sequence)

        loss = self.criterion(logits, targets)

        log = {
            'train_loss': loss.item(),
            'train_perplexity_ls': np.exp(loss.item()),
            'batch_len': source_sequence.size(0) * source_sequence.size(1),
            'training_tokens': targets.size(0)
        }

        return {'loss': loss, 'log': log}

    def validation_step(self, batch, batch_idx):
        source_sequence, target_sequence, segment_indices, position_indices = batch

        self.model.set_seq2seq()

        logits, targets = self.forward(source_sequence, segment_indices, position_indices, target_sequence)

        validation_loss = self.criterion(logits, targets)

        return {'val_loss': validation_loss}

    def validation_epoch_end(self, outputs):
        mean_loss = torch.stack([x['val_loss'] for x in outputs]).mean().item()

        log = {
            'val_loss': mean_loss,
            'val_perplexity': np.exp(mean_loss)
        }

        return {'val_loss': mean_loss, 'log': log}
