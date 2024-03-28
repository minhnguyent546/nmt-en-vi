import sys
import argparse
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn

from tokenizers import Tokenizer

from nmt.utils import (
    model as model_util,
    config as config_util,
    bleu as bleu_util,
)
from nmt.utils.misc import set_seed
import nmt.constants as const

def test_model(config: dict):
    set_seed(config['seed'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    checkpoints_dir = Path(config['checkpoints_dir'])
    model_dir = checkpoints_dir / config['model_dir']
    model_dir.mkdir(parents=True, exist_ok=True)

    print('Loading data loaders')
    data_loaders = torch.load(checkpoints_dir / config['data_loaders_basename'])
    test_data_loader = data_loaders['test']

    print('Loading tokenizers')
    src_tokenizer = Tokenizer.from_file(str(checkpoints_dir / config['tokenizer_basename'].format(config['src_lang'])))
    target_tokenizer = Tokenizer.from_file(str(checkpoints_dir / config['tokenizer_basename'].format(config['target_lang'])))

    model = model_util.make_model(src_tokenizer, target_tokenizer, config)
    model.to(device)

    test_checkpoint = config['test_checkpoint']
    model_weights_path = None
    if test_checkpoint == const.LATEST:
        model_weights_path = model_util.get_latest_weights_file_path(config=config)
    else:
        model_weights_path = model_util.get_weights_file_path(f'{test_checkpoint:0>2}', config)

    if model_weights_path is None:
        print('No model weights found to load')
        print('Aborted')
        sys.exit(1)

    print(f'Loading weights from checkpoint: {test_checkpoint}')

    states = torch.load(model_weights_path, map_location=device)

    model.load_state_dict(states['model_state_dict'])

    loss_function = nn.CrossEntropyLoss(ignore_index=src_tokenizer.token_to_id(const.PAD_TOKEN),
                                        label_smoothing=config['label_smoothing'])

    test_stats = model_util.evaluate(model, loss_function, test_data_loader)
    test_bleu = bleu_util.compute_dataset_bleu(model, test_data_loader.dataset,
                                              target_tokenizer, config['seq_length'],
                                              **config['compute_bleu_kwargs'], )
    print(pd.DataFrame({
        'test_loss': [test_stats['eval_loss']],
        'test_accuracy': [test_stats['eval_accuracy']],
        'val_bleu-1': [test_bleu[0]],
        'val_bleu-2': [test_bleu[1]],
        'val_bleu-3': [test_bleu[2]],
        'val_bleu-4': [test_bleu[3]],
    }).to_string(index=False))

def main():
    parser = argparse.ArgumentParser(description='Test the model')
    parser.add_argument('--config',
                        help='Path to the config file (default: ./config/config.yaml)',
                        dest='config_file',
                        default='./config/config.yaml')

    args = parser.parse_args()
    config = config_util.get_config(args.config_file)
    test_model(config)


if __name__ == '__main__':
    main()
