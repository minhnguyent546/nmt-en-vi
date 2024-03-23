import argparse
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from tokenizers import Tokenizer

import utils.model as model_util
import utils.config as config_util
import utils.bleu as bleu_util
import constants as const

def train_model(config: dict):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device {device}')
    device = torch.device(device)

    checkpoints_dir = Path(config['checkpoints_dir'])
    model_dir = checkpoints_dir / config['model_dir']
    model_dir.mkdir(parents=True, exist_ok=True)

    print('Loading data loaders')
    data_loaders = torch.load(checkpoints_dir / config['data_loaders_basename'])
    train_data_loader = data_loaders['train']
    validation_data_loader = data_loaders['validation']

    print('Loading tokenizers')
    src_tokenizer = Tokenizer.from_file(str(checkpoints_dir / config['tokenizer_basename'].format(config['src_lang'])))
    target_tokenizer = Tokenizer.from_file(str(checkpoints_dir / config['tokenizer_basename'].format(config['target_lang'])))

    model = model_util.make_model(src_tokenizer, target_tokenizer, config)
    model.to(device)

    # Tensorboard
    writer = SummaryWriter(config['experiment_name'])

    # optimizer and lr scheduler
    learning_rate = config['learning_rate']
    optimizer = model_util.make_optimizer(model, config)
    lr_scheduler = None
    if config['enable_lr_scheduler']:
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step_num: learning_rate
                                       * model_util.noam_decay(
                                            step_num,
                                            d_model=config['d_model'],
                                            warmup_steps=config['warmup_steps']
                                        )
        )

    initial_epoch = 0
    global_step = 0
    preload = config['preload']
    if preload is not None:
        model_filename = None
        if preload == 'latest':
            model_filename = model_util.get_latest_weights_file_path(config=config)
        else:
            model_filename = model_util.get_weights_file_path(f'{preload:0>2}', config=config)

        if model_filename is None:
            print('No model weights found to preload')
        else:
            print(f'Loading weights from previous epoch: {preload:0>2}')
            states = torch.load(model_filename, map_location=device)

            # continue from previous completed epoch
            initial_epoch = states['epoch'] + 1

            model.load_state_dict(states['model_state_dict'])
            optimizer.load_state_dict(states['optimizer_state_dict'])
            if lr_scheduler is not None and 'lr_scheduler_state_dict' in states:
                lr_scheduler.load_state_dict(states['lr_scheduler_state_dict'])
            global_step = states['global_step']

    loss_function = nn.CrossEntropyLoss(ignore_index=src_tokenizer.token_to_id(const.PAD_TOKEN),
                                        label_smoothing=config['label_smoothing'])

    num_epochs = config['num_epochs']
    for epoch in range(initial_epoch, num_epochs):
        # clear cuda cache
        torch.cuda.empty_cache()

        train_stats = model_util.train(model, device, optimizer, loss_function,
                                       train_data_loader, epoch, global_step,
                                       config, train_max_steps=config['per_epoch_train_max_steps'],
                                       writer=writer, lr_scheduler=lr_scheduler)

        val_stats = model_util.evaluate(model, device, loss_function,
                                        validation_data_loader,
                                        eval_max_steps=config['val_max_steps'])

        val_bleu = bleu_util.compute_dataset_bleu(model, device, validation_data_loader.dataset,
                                                  target_tokenizer, config['seq_length'],
                                                  teacher_forcing=False,
                                                  beam_size=config['beam_size'],
                                                  beam_return_topk=config['beam_return_topk'],
                                                  max_n=4, log_sentences=True,
                                                  logging_interval=20, max_steps=200)

        writer.add_scalars('loss', {
            'train_loss': train_stats['train_loss'],
            'val_loss': val_stats['eval_loss'],
        }, epoch)
        writer.add_scalars('accuracy', {
            'train_accuracy': train_stats['train_accuracy'],
            'val_accuracy': val_stats['eval_accuracy'],
        }, epoch)
        writer.add_scalars('bleu/val_bleu', {
            f'BLEU-{i + 1}': val_bleu[i]
            for i in range(4)
        }, epoch)

        # write epoch information to the screen
        print(pd.DataFrame({
            'epoch': [epoch + 1],
            'global_step': [train_stats['global_step']],
            'train_loss': [train_stats['train_loss']],
            'val_loss': [val_stats['eval_loss']],
            'train_accuracy': [train_stats['train_accuracy']],
            'val_accuracy': [val_stats['eval_accuracy']],
            'val_bleu-1': [val_bleu[0]],
            'val_bleu-2': [val_bleu[1]],
            'val_bleu-3': [val_bleu[2]],
            'val_bleu-4': [val_bleu[3]],
        }).to_string(index=False))

        global_step = train_stats['global_step']

        # save the model after every epoch
        model_checkpoint_path = model_util.get_weights_file_path(f'{epoch:02d}', config)
        checkpoint_dict = {
            'epoch': epoch,
            'global_step': global_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        if lr_scheduler is not None:
            checkpoint_dict['lr_scheduler_state_dict'] = lr_scheduler.state_dict()

        torch.save(checkpoint_dict, model_checkpoint_path)

def main(config_file: str):
    config = config_util.get_config(config_file)
    train_model(config)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the model')
    parser.add_argument('--config',
                        help='Path to the config file (default: ./config/config.yaml)',
                        dest='config_file',
                        default='./config/config.yaml')

    args = parser.parse_args()
    main(**vars((args)))
