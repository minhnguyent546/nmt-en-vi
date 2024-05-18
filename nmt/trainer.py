from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any

from tqdm.autonotebook import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from tokenizers import Tokenizer

from transformer import Transformer, TransformerConfig
from nmt.utils import (
    model as model_util,
    bleu as bleu_util,
)
from nmt.utils import stats
from nmt.utils.logging import logger


@dataclass
class TrainingArguments():
    model_save_dir: str
    model_basename: str = 'transformer'
    saved_checkpoints_limit: int = 6
    train_steps: int = 50_000
    valid_interval: int = 3_000
    valid_compute_bleu_kwargs: dict[str, Any] = field(default_factory=dict)
    save_every: int = 5_000
    train_batch_size: int = 32
    eval_batch_size: int = 32
    fp16: bool = False
    label_smoothing: float = 0.0
    max_grad_norm: float = 0.0
    initial_global_step: int = 0
    initial_train_stats: stats.Stats | None = None

class Trainer:
    def __init__(
        self,
        model: Transformer,
        optimizer: torch.optim.Optimizer,
        criterion: nn.CrossEntropyLoss,
        src_tokenizer: Tokenizer,
        target_tokenizer: Tokenizer,
        args: TrainingArguments,
        transformer_config: TransformerConfig,
        lr_scheduler=None,
        scaler_state_dict=None,
        writer: SummaryWriter | None = None,
    ) -> None:
        self.model = model
        self.device = model.device
        self.optimizer = optimizer
        self.criterion = criterion
        self.src_tokenizer = src_tokenizer
        self.target_tokenizer = target_tokenizer
        self.args = args
        self.transformer_config = transformer_config
        self.lr_scheduler = lr_scheduler
        self.writer = writer
        if args.initial_train_stats is not None:
            self.train_stats = args.initial_train_stats
        else:
            self.train_stats = stats.Stats(ignore_padding=True, pad_token_id=self.model.target_pad_token_id)

        # mixed precision training with fp16
        self.autocast_ctx = nullcontext()
        self.train_dtype = torch.float32
        if args.fp16 and torch.cuda.is_available() and self.model.device.type == 'cuda':
            self.train_dtype = torch.float16
            self.autocast_ctx = torch.cuda.amp.autocast(dtype=self.train_dtype)
            logger.warning(
                f'Mixed precision training enabled with dtype: {self.train_dtype}. '
                'When mixed precision is enabled, the training process can not be resumed from previous checkpoints. '
                'See related issue here: https://discuss.pytorch.org/t/resume-training-with-mixed-precision-lead-to-no-inf-checks-were-recorded-for-this-optimizer/115828/3'
            )
        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.train_dtype == torch.float16))
        if scaler_state_dict is not None:
            self.scaler.load_state_dict(scaler_state_dict)

    def train(
        self,
        train_data_loader: DataLoader,
        validation_data_loader: DataLoader,
    ) -> None:
        # set model in training mode
        self.model.train()

        global_step = self.args.initial_global_step
        train_progress_bar = tqdm(range(global_step, self.args.train_steps), desc='Training model')
        while global_step < self.args.train_steps:
            # empty cuda cache
            torch.cuda.empty_cache()

            # TODO: currently, this training does not care about the state of the data loader
            for batch in train_data_loader:
                encoder_input = batch['encoder_input'].to(self.device)  # (batch_size, seq_length)
                decoder_input = batch['decoder_input'].to(self.device)  # (batch_size, seq_length)

                self.optimizer.zero_grad()

                with self.autocast_ctx:
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
                self.scaler.scale(loss).backward()

                if self.args.max_grad_norm > 0:
                    # clipping the gradient
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.args.max_grad_norm)

                # update weights and learning rate
                self.scaler.step(self.optimizer)
                self.scaler.update()

                if self.lr_scheduler is not None:
                    if self.writer is not None:
                        for group_id, group_lr in enumerate(self.lr_scheduler.get_last_lr()):
                            self.writer.add_scalar(f'learning_rate/group-{group_id}', group_lr, global_step)
                    self.lr_scheduler.step()

                self.train_stats.update_step(loss.item(), pred.view(-1), labels.view(-1))
                train_progress_bar.set_postfix({'loss': f'{loss.item():0.3f}'})

                if self.writer is not None:
                    self.writer.add_scalar('loss/train_batch_loss', loss.item(), global_step)

                    if (global_step + 1) % self.args.valid_interval == 0:
                        valid_stats = model_util.evaluate(self.model, self.criterion, validation_data_loader)
                        valid_bleu = bleu_util.compute_dataset_bleu(
                            self.model,
                            validation_data_loader.dataset,
                            self.src_tokenizer,
                            self.target_tokenizer,
                            self.transformer_config.target_seq_length,
                            **self.args.valid_compute_bleu_kwargs
                        )
                        self._report_stats(global_step + 1, valid_stats, valid_bleu)

                    self.writer.flush()

                if (global_step + 1) % self.args.save_every == 0:
                    self._save_checkpoint(global_step + 1)

                global_step += 1
                train_progress_bar.update()
                if global_step >= self.args.train_steps:
                    break

    def _report_stats(
        self,
        step: int,
        valid_stats: stats.Stats,
        valid_bleu: float | None = None,
    ) -> None:
        if self.writer is None:
            return

        self.train_stats.report_to_tensorboard(self.writer, name='train', step=step)
        valid_stats.report_to_tensorboard(self.writer, name='valid', step=step)

        if valid_bleu is not None:
            self.writer.add_scalar('valid_BLEU', valid_bleu, step)

        # reset train statistics after reporting
        self.train_stats = stats.Stats(ignore_padding=True, pad_token_id=self.model.target_pad_token_id)

    def _save_checkpoint(self, step: int) -> None:
        model_checkpoint_path = model_util.get_weights_file_path(
            self.args.model_save_dir,
            self.args.model_basename,
            step,
        )
        checkpoint_dict = {
            'global_step': step,
            'train_stats': self.train_stats,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'config': self.transformer_config,
        }

        if self.lr_scheduler is not None:
            checkpoint_dict['lr_scheduler_state_dict'] = self.lr_scheduler.state_dict()

        model_util.ensure_num_saved_checkpoints(
            self.args.model_save_dir,
            self.args.model_basename,
            self.args.saved_checkpoints_limit - 1,
        )
        torch.save(checkpoint_dict, model_checkpoint_path)
