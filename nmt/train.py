import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from datasets import load_from_disk, DatasetDict

from nmt.constants import SpecialToken
from nmt.trainer import Trainer, TrainingArguments
from nmt.utils import (
    model as model_util,
    config as config_util,
    dataset as dataset_util,
)
from nmt.utils.logging import init_logger, logger
from nmt.utils.misc import set_seed
from transformer import build_transformer, TransformerConfig


def train_model(config: dict):
    set_seed(config['seed'])
    init_logger()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info('Using device %s', device)
    device = torch.device(device)

    checkpoints_dir = Path(config['checkpoints_dir'])
    model_dir = checkpoints_dir / config['model_dir']
    model_dir.mkdir(parents=True, exist_ok=True)

    logger.info('Loading tokenizers')
    src_tokenizer, target_tokenizer = dataset_util.load_trained_tokenizers(config)

    logger.info('Creating data loaders')
    saved_dataset: DatasetDict = load_from_disk(config['dataset_save_path'])
    train_data_loader = dataset_util.make_data_loader(
        saved_dataset['train'],
        src_tokenizer,
        target_tokenizer,
        batch_size=config['train_batch_size'],
        shuffle=True,
        config=config
    )
    validation_data_loader = dataset_util.make_data_loader(
        saved_dataset['validation'],
        src_tokenizer,
        target_tokenizer,
        batch_size=config['eval_batch_size'],
        shuffle=False,
        config=config
    )

    transformer_config = TransformerConfig(
        src_vocab_size=src_tokenizer.get_vocab_size(),
        target_vocab_size=target_tokenizer.get_vocab_size(),
        src_seq_length=config['src_seq_length'],
        target_seq_length=config['target_seq_length'],
        src_pad_token_id=src_tokenizer.token_to_id(SpecialToken.PAD),
        target_pad_token_id=target_tokenizer.token_to_id(SpecialToken.PAD),
        device=device,
        d_model=config['d_model'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        d_ffn=config['d_ffn'],
        dropout=config['dropout'],
        attention_dropout=config['attention_dropout'],
    )
    model = build_transformer(transformer_config)
    model.to(device)

    # optimizer and lr scheduler
    learning_rate = config['learning_rate']
    optimizer = model_util.make_optimizer(model, config)
    lr_scheduler = None
    if config['enable_lr_scheduler']:
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step_num: learning_rate * model_util.noam_decay(
                step_num,
                d_model=config['d_model'],
                warmup_steps=config['warmup_steps'])
        )

    initial_global_step = 0
    initial_train_stats = None
    from_checkpoint = config['from_checkpoint']
    scaler_state_dict = None
    if from_checkpoint is not None:
        logger.info('Loading states from checkpoint: %s', from_checkpoint)
        checkpoint_states = torch.load(from_checkpoint, map_location=device)
        required_keys = [
            'model_state_dict',
            'optimizer_state_dict',
            'config',
        ]
        if lr_scheduler is not None:
            required_keys.append('lr_scheduler_state_dict')
        for key in required_keys:
            if key not in checkpoint_states:
                raise ValueError(f'Missing key "{key}" in checkpoint')

        optimizer.load_state_dict(checkpoint_states['optimizer_state_dict'])
        transformer_config = checkpoint_states['config']
        model = build_transformer(transformer_config).to(device)
        model.load_state_dict(checkpoint_states['model_state_dict'])
        if 'scaler_state_dict' in checkpoint_states:
            scaler_state_dict = checkpoint_states['scaler_state_dict']

        if lr_scheduler is not None:
            lr_scheduler.load_state_dict(checkpoint_states['lr_scheduler_state_dict'])

        initial_global_step = checkpoint_states.get('global_step', initial_global_step)
        initial_train_stats = checkpoint_states.get('train_stats', initial_train_stats)

    criterion = nn.CrossEntropyLoss(ignore_index=src_tokenizer.token_to_id(SpecialToken.PAD),
                                    label_smoothing=config['label_smoothing'])

    # Tensorboard
    writer = SummaryWriter(config['experiment_name'])

    training_args = TrainingArguments(
        model_save_dir=str(Path(config['checkpoints_dir']) / config['model_dir']),
        model_basename=config['model_basename'],
        saved_checkpoints_limit=config['saved_checkpoints_limit'],
        train_steps=config['train_steps'],
        valid_interval=config['valid_interval'],
        valid_compute_bleu_kwargs=config['valid_compute_bleu_kwargs'],
        save_every=config['save_every'],
        train_batch_size=config['train_batch_size'],
        eval_batch_size=config['eval_batch_size'],
        fp16=config['fp16'],
        label_smoothing=config['label_smoothing'],
        max_grad_norm=config['max_grad_norm'],
        initial_global_step=initial_global_step,
        initial_train_stats=initial_train_stats,
    )
    trainer = Trainer(
        model,
        optimizer,
        criterion,
        src_tokenizer,
        target_tokenizer,
        training_args,
        transformer_config,
        lr_scheduler=lr_scheduler,
        writer=writer,
        scaler_state_dict=scaler_state_dict,
    )
    trainer.train(train_data_loader, validation_data_loader)

def main():
    parser = argparse.ArgumentParser(description='Train the model')
    parser.add_argument('--config',
                        help='Path to the config file (default: ./config/config.yaml)',
                        dest='config_file',
                        default='./config/config.yaml')

    args = parser.parse_args()
    config = config_util.get_config(args.config_file)
    train_model(config)


if __name__ == '__main__':
    main()
