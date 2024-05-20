import argparse
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn

from datasets import DatasetDict, load_from_disk

from nmt.constants import SpecialToken
from nmt.utils import (
    bleu as bleu_util,
    config as config_util,
    dataset as dataset_util,
    model as model_util,
)
from nmt.utils.logging import init_logger, logger
from nmt.utils.misc import set_seed
from transformer import build_transformer


def test_model(config: dict):
    set_seed(config['seed'])
    init_logger()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    checkpoints_dir = Path(config['checkpoints_dir'])
    model_dir = checkpoints_dir / config['model_dir']
    model_dir.mkdir(parents=True, exist_ok=True)

    logger.info('Loading tokenizers')
    src_tokenizer, target_tokenizer = dataset_util.load_trained_tokenizers(config)

    logger.info('Creating data loader')
    saved_dataset: DatasetDict = load_from_disk(config['dataset_save_path'])
    test_data_loader = dataset_util.make_data_loader(
        saved_dataset['test'],
        src_tokenizer,
        target_tokenizer,
        batch_size=config['eval_batch_size'],
        shuffle=False,
        config=config,
    )

    test_checkpoint = config['test_checkpoint']
    logger.info('Testing model with checkpoint: %s', test_checkpoint)
    checkpoint_states = torch.load(test_checkpoint, map_location=device)
    required_keys = ['config', 'model_state_dict']
    for key in required_keys:
        if key not in checkpoint_states:
            raise ValueError(f'Missing key "{key}" in checkpoint')
    transformer_config = checkpoint_states['config']
    model = build_transformer(transformer_config).to(device)
    model.load_state_dict(checkpoint_states['model_state_dict'])

    criterion = nn.CrossEntropyLoss(ignore_index=src_tokenizer.token_to_id(SpecialToken.PAD),
                                    label_smoothing=config['label_smoothing'])

    test_stats = model_util.evaluate(model, criterion, test_data_loader)
    test_bleu = bleu_util.compute_dataset_bleu(model, test_data_loader.dataset,
                                               src_tokenizer, target_tokenizer,
                                               config['target_seq_length'],
                                               **config['valid_compute_bleu_kwargs'])

    metric_scores = test_stats.compute()

    print(pd.DataFrame({
        'test_loss': [metric_scores['loss']],
        'test_accuracy': [metric_scores['acc']],
        'test_precision': [metric_scores['precision']],
        'test_recall': [metric_scores['recall']],
        'test_f_beta': [metric_scores['f_beta']],
        'test_bleu': [test_bleu],
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
