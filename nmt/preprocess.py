from pathlib import Path
import argparse
import pandas as pd

import torch
from torch.utils.data import DataLoader

from datasets import load_dataset, Dataset, DatasetDict
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from nmt.billingual_dataset import BilingualDataset
from nmt.utils import (
    dataset as dataset_util,
    config as config_util,
)
from nmt.utils.misc import set_seed
from nmt.constants import SpecialToken

def tokenize(dataset: Dataset, lang: str, config: dict, min_freq: int = 2) -> Tokenizer:
    checkpoints_dir = Path(config['checkpoints_dir'])
    tokenizer_path = checkpoints_dir / config['tokenizer_basename'].format(lang)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = Tokenizer(WordLevel(unk_token=SpecialToken.UNK))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = WordLevelTrainer(
        min_frequency=min_freq,
        special_tokens=[SpecialToken.PAD, SpecialToken.SOS, SpecialToken.EOS, SpecialToken.UNK]
    )
    dataset_iter = dataset_util.create_iter_from_dataset(dataset, lang)
    tokenizer.train_from_iterator(dataset_iter, trainer=trainer)
    tokenizer.save(str(tokenizer_path))

    return tokenizer

def _load_datasets(config: dict) -> DatasetDict | Dataset:
    raw_datasets: DatasetDict = load_dataset(
        path=config['dataset_path'],
        name=config['dataset_name'],
        cache_dir=config['dataset_cache_dir'],
        **config['dataset_other_options']
    )

    # creating validation set from train set
    if config['val_size_rate'] is not None:
        old_datasets = raw_datasets
        raw_datasets = old_datasets['train'].train_test_split(test_size=config['val_size_rate'])
        # rename the default "test" split to "validation"
        raw_datasets['validation'] = raw_datasets.pop('test')
        # add the test set for raw_datasets
        raw_datasets['test'] = old_datasets['test']

    # slicing the dataset
    for split in raw_datasets:
        if split in config['max_set_size'] and config['max_set_size'][split] is not None:
            max_set_size = config['max_set_size'][split]
            if 0 < max_set_size and max_set_size < len(raw_datasets[split]):
                raw_datasets[split] = raw_datasets[split].shuffle().select(range(max_set_size))

    return raw_datasets

def preprocess(config: dict):
    set_seed(config['seed'])
    raw_datasets = _load_datasets(config)
    num_rows = raw_datasets.num_rows
    raw_datasets = dataset_util.process_dataset_sentences(raw_datasets,
                                                          langs=[config['src_lang'], config['target_lang']],
                                                          vi_config=config,
                                                          batched=True)

    print(pd.DataFrame(raw_datasets['train']['translation'][:5]))
    print(pd.DataFrame(raw_datasets['validation']['translation'][:5]))
    print(pd.DataFrame(raw_datasets['test']['translation'][:5]))

    print('Building tokenizers from train dataset')
    src_tokenizer = tokenize(raw_datasets['train'], config['src_lang'], config)
    target_tokenizer = tokenize(raw_datasets['train'], config['target_lang'], config)
    print('Size of source vocabulary:', src_tokenizer.get_vocab_size())
    print('Size of target vocabulary:', target_tokenizer.get_vocab_size())

    print('Removing invalid sentences')
    num_reserved_tokens = 2  # for SOS and EOS tokens
    raw_datasets = dataset_util.remove_invalid_sentences(raw_datasets,
                                                         src_tokenizer,
                                                         target_tokenizer,
                                                         config['seq_length'] - num_reserved_tokens,
                                                         config['src_lang'],
                                                         config['target_lang'],
                                                         batched=True)

    for dataset, num_row in raw_datasets.num_rows.items():
        if dataset in num_rows:
            print(f'Removed {num_rows[dataset] - num_row} sentences from {dataset}')

    train_dataset = BilingualDataset(
        raw_datasets['train'],
        src_tokenizer,
        target_tokenizer,
        config['src_lang'],
        config['target_lang'],
        config['seq_length']
    )
    validation_dataset = BilingualDataset(
        raw_datasets['validation'],
        src_tokenizer,
        target_tokenizer,
        config['src_lang'],
        config['target_lang'],
        config['seq_length']
    )
    test_dataset = BilingualDataset(
        raw_datasets['test'],
        src_tokenizer,
        target_tokenizer,
        config['src_lang'],
        config['target_lang'],
        config['seq_length']
    )

    assert src_tokenizer.token_to_id(SpecialToken.PAD) == target_tokenizer.token_to_id(SpecialToken.PAD)
    pad_token_id = src_tokenizer.token_to_id(SpecialToken.PAD)
    data_collator = dataset_util.CollatorWithPadding(
        pad_token_id,
        added_features=['encoder_input', 'decoder_input', 'labels']
    )
    train_data_loader = DataLoader(train_dataset, batch_size=config['train_batch_size'],
                                   shuffle=True, collate_fn=data_collator)
    validation_data_loader = DataLoader(validation_dataset, batch_size=config['eval_batch_size'],
                                        collate_fn=data_collator)
    test_data_loader = DataLoader(test_dataset, batch_size=config['eval_batch_size'],
                                  collate_fn=data_collator)
    data_loaders = {
        'train': train_data_loader,
        'validation': validation_data_loader,
        'test': test_data_loader,
    }
    checkpoints_dir = Path(config['checkpoints_dir'])
    data_loaders_path = checkpoints_dir / config['data_loaders_basename']
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    torch.save(data_loaders, data_loaders_path)

def main():
    parser = argparse.ArgumentParser(description='Preprocess the dataset')
    parser.add_argument('--config',
                        help='Path to the config file (default: ./config/config.yaml)',
                        dest='config_file',
                        default='./config/config.yaml')

    args = parser.parse_args()
    config = config_util.get_config(args.config_file)
    preprocess(config)


if __name__ == '__main__':
    main()
