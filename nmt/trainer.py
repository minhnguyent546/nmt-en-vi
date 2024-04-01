from tqdm.autonotebook import tqdm

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from tokenizers import Tokenizer

from transformer import Transformer
from nmt.utils import (
    model as model_util,
    bleu as bleu_util,
)
from nmt.utils import stats


class Trainer:
    def __init__(
        self,
        model: Transformer,
        optimizer: torch.optim.Optimizer,
        criterion,
        src_tokenizer: Tokenizer,
        target_tokenizer: Tokenizer,
        config: dict,
        writer: SummaryWriter | None = None,
        lr_scheduler=None,
    ) -> None:

        self.model = model
        self.device = model.device
        self.optimizer = optimizer
        self.criterion = criterion
        self.src_tokenizer = src_tokenizer
        self.target_tokenizer = target_tokenizer
        self.config = config
        self.initial_epoch = 0
        self.global_step = 0
        self.train_stats = stats.Stats()
        self.writer = writer
        self.lr_scheduler = lr_scheduler

    def train(
        self,
        train_data_loader: DataLoader,
        validation_data_loader: DataLoader,
        validation_interval: int = 3000,
        preload_states=None,
    ) -> None:

        if preload_states is not None:
            self._load_from_states(preload_states)

        # set model in training mode
        self.model.train()

        num_epochs = self.config['num_epochs']
        for epoch in range(self.initial_epoch, num_epochs):
            # clear cuda cache
            torch.cuda.empty_cache()

            batch_iterator = tqdm(train_data_loader,
                                  desc=f'Processing epoch {epoch + 1:02d}/{num_epochs:02d}')
            for batch in batch_iterator:
                encoder_input = batch['encoder_input'].to(self.device)  # (batch_size, seq_length)
                decoder_input = batch['decoder_input'].to(self.device)  # (batch_size, seq_length)

                decoder_output = self.model(encoder_input, decoder_input)  # (batch_size, seq_length, d_model)
                logits = self.model.linear(decoder_output)  # (batch_size, seq_length, target_vocab_size)
                pred = logits.argmax(dim=-1)  # (batch_size, seq_length)
                labels = batch['labels'].to(self.device)  # (batch_size, seq_length)

                # calculate the loss
                # logits: (batch_size * seq_length, target_vocab_size)
                # label: (batch_size * seq_length)
                target_vocab_size = logits.size(-1)
                loss = self.criterion(logits.view(-1, target_vocab_size), labels.view(-1))

                # backpropagate the loss
                loss.backward()

                if self.config['max_grad_norm'] > 0:
                    # clipping the gradient
                    nn.utils.clip_grad_norm_(self.model.parameters(),
                                             max_norm=self.config['max_grad_norm'])

                # update weights and learning rate
                self.optimizer.step()
                self.optimizer.zero_grad()
                if self.lr_scheduler is not None:
                    if self.writer is not None:
                        for group_id, group_lr in enumerate(self.lr_scheduler.get_last_lr()):
                            self.writer.add_scalar(f'learning_rate/group-{group_id}', group_lr, self.global_step)
                    self.lr_scheduler.step()

                self.train_stats.update_step(loss.item(), pred.view(-1), labels.view(-1))
                batch_iterator.set_postfix({'loss': f'{loss.item():0.3f}'})

                if self.writer is not None:
                    self.writer.add_scalar('loss/train_batch_loss', loss.item(), self.global_step)

                    if (self.global_step + 1) % validation_interval == 0:
                        valid_stats = model_util.evaluate(self.model, self.criterion, validation_data_loader)
                        valid_bleu = bleu_util.compute_dataset_bleu(self.model,
                                                                    validation_data_loader.dataset,
                                                                    self.target_tokenizer,
                                                                    self.config['seq_length'],
                                                                    **self.config['compute_bleu_kwargs'])
                        self._report_stats(self.global_step + 1, valid_stats, valid_bleu)

                    self.writer.flush()

                self.global_step += 1

            self._save_checkpoint(epoch)

    def _load_from_states(self, states) -> None:
        self.initial_epoch = states['epoch'] + 1
        self.global_step = states['global_step']
        self.train_stats = states['train_stats']
        self.model.load_state_dict(states['model_state_dict'])
        self.optimizer.load_state_dict(states['optimizer_state_dict'])
        if self.lr_scheduler is not None and 'lr_scheduler_state_dict' in states:
            self.lr_scheduler.load_state_dict(states['lr_scheduler_state_dict'])

    def _report_stats(
        self,
        step: int,
        valid_stats: stats.Stats,
        valid_bleu: list[float] | None = None,
    ) -> None:
        if self.writer is None:
            return

        self.train_stats.report_to_tensorboard(self.writer, name='train', step=step)
        valid_stats.report_to_tensorboard(self.writer, name='valid', step=step)

        if valid_bleu is not None:
            self.writer.add_scalars('valid_bleu', {
                f'BLEU-{i + 1}': valid_bleu[i]
                for i in range(4)
            }, step)

        self.train_stats = stats.Stats()

    def _save_checkpoint(self, epoch: int) -> None:
        model_checkpoint_path = model_util.get_weights_file_path(f'{epoch:02d}', self.config)
        checkpoint_dict = {
            'epoch': epoch,
            'global_step': self.global_step,
            'train_stats': self.train_stats,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }

        if self.lr_scheduler is not None:
            checkpoint_dict['lr_scheduler_state_dict'] = self.lr_scheduler.state_dict()

        torch.save(checkpoint_dict, model_checkpoint_path)
