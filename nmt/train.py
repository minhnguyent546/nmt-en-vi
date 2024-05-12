import argparse

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from datasets import load_from_disk, DatasetDict

from nmt.utils import (
    model as model_util,
    config as config_util,
    dataset as dataset_util,
)
from nmt.utils.misc import set_seed
from nmt.trainer import Trainer
from nmt.constants import SpecialToken, Epoch


def train_model(config: dict):
    set_seed(config['seed'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device {device}')
    device = torch.device(device)

    print('Loading tokenizers')
    src_tokenizer, target_tokenizer = dataset_util.load_trained_tokenizers(config)

    print('Creating data loaders')
    saved_dataset: DatasetDict = load_from_disk(config['dataset_save_path'])
    train_data_loader = dataset_util.make_data_loader(saved_dataset['train'],
                                                      src_tokenizer,
                                                      target_tokenizer,
                                                      batch_size=config['train_batch_size'],
                                                      shuffle=True,
                                                      config=config)
    validation_data_loader = dataset_util.make_data_loader(saved_dataset['validation'],
                                                           src_tokenizer,
                                                           target_tokenizer,
                                                           batch_size=config['eval_batch_size'],
                                                           shuffle=False,
                                                           config=config)

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

    preload = config['preload']
    preload_states = None
    if preload is not None:
        model_filename = None
        if preload == Epoch.LATEST:
            model_filename = model_util.get_latest_weights_file_path(config=config)
        else:
            model_filename = model_util.get_weights_file_path(f'{preload:0>2}', config=config)

        if model_filename is not None:
            print('Load states from previous process')
            preload_states = torch.load(model_filename)

    criterion = nn.CrossEntropyLoss(ignore_index=src_tokenizer.token_to_id(SpecialToken.PAD),
                                    label_smoothing=config['label_smoothing'])

    trainer = Trainer(model,
                      optimizer,
                      criterion,
                      src_tokenizer,
                      target_tokenizer,
                      config,
                      writer=writer,
                      lr_scheduler=lr_scheduler)
    trainer.train(train_data_loader,
                  validation_data_loader,
                  validation_interval=config['validation_interval'],
                  preload_states=preload_states)

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
