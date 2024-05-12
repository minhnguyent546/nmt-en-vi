import sys
import argparse
import pandas as pd

import torch
import torch.nn as nn

from datasets import load_from_disk, DatasetDict

from nmt.utils import (
    model as model_util,
    config as config_util,
    bleu as bleu_util,
    dataset as dataset_util,
)
from nmt.utils.misc import set_seed
from nmt.constants import SpecialToken, Epoch

def test_model(config: dict):
    set_seed(config['seed'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    print('Loading tokenizers')
    src_tokenizer, target_tokenizer = dataset_util.load_trained_tokenizers(config)

    print('Creating data loader')
    saved_dataset: DatasetDict = load_from_disk(config['dataset_save_path'])
    test_data_loader = dataset_util.make_data_loader(saved_dataset['test'],
                                                     src_tokenizer,
                                                     target_tokenizer,
                                                     batch_size=config['eval_batch_size'],
                                                     shuffle=False,
                                                     config=config)

    model = model_util.make_model(src_tokenizer, target_tokenizer, config)
    model.to(device)

    test_checkpoint = config['test_checkpoint']
    model_weights_path = None
    if test_checkpoint == Epoch.LATEST:
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

    criterion = nn.CrossEntropyLoss(ignore_index=src_tokenizer.token_to_id(SpecialToken.PAD),
                                    label_smoothing=config['label_smoothing'])

    test_stats = model_util.evaluate(model, criterion, test_data_loader)
    test_bleu = bleu_util.compute_dataset_bleu(model, test_data_loader.dataset,
                                               src_tokenizer, target_tokenizer,
                                               config['seq_length'],
                                               **config['compute_bleu_kwargs'])

    metric_scores = test_stats.compute()

    print(pd.DataFrame({
        'test_loss': [metric_scores['loss']],
        'test_accuracy': [metric_scores['acc']],
        'test_precision': [metric_scores['precision']],
        'test_recall': [metric_scores['recall']],
        'test_f_beta': [metric_scores['f_beta']],
        'test_blue': [test_bleu],
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
