from pathlib import Path
import pandas as pd

import torch
from torch.utils.data import DataLoader

from datasets import load_dataset, Dataset, DatasetDict
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from billingual_dataset import BilingualDataset
import utils.dataset as dataset_util
import utils.config as config_util

import constants as const

def tokenize(dataset: Dataset, lang: str, config: dict, min_freq: int = 2) -> Tokenizer:
    checkpoints_dir = Path(config['checkpoints_dir'])
    tokenizer_path = checkpoints_dir / config['tokenizer_basename'].format(lang)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer(WordLevel(unk_token=const.UNK_TOKEN))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = WordLevelTrainer(
        min_frequency=min_freq,
        special_tokens=[const.PAD_TOKEN, const.SOS_TOKEN, const.EOS_TOKEN, const.UNK_TOKEN]
    )
    dataset_iter = dataset_util.create_iter_from_dataset(dataset, lang)
    tokenizer.train_from_iterator(dataset_iter, trainer=trainer)
    tokenizer.save(str(tokenizer_path))

    return tokenizer

def preprocess(config: dict):
    dataset_dict: DatasetDict = load_dataset(
        path=config['dataset_path'],
        name=config['dataset_subset'],
        cache_dir=config['dataset_cache_dir'],
    )
    raw_datasets = dataset_dict['train'].train_test_split(test_size=0.1, seed=config['seed'])
    # rename the default "test" split to "validation"
    raw_datasets['validation'] = raw_datasets.pop('test')
    # add the test set for raw_datasets
    raw_datasets['test'] = dataset_dict['test']

    num_rows = raw_datasets.num_rows
    raw_datasets = dataset_util.process_dataset_sentences(raw_datasets,
                                                          langs=[config['src_lang'], config['target_lang']],
                                                          vi_config=config,
                                                          batched=True)

    print(pd.DataFrame(raw_datasets['train']['translation'][:25]))

    print('Building tokenizers from train dataset')
    src_tokenizer = tokenize(raw_datasets['train'], config['src_lang'], config, min_freq=2)
    target_tokenizer = tokenize(raw_datasets['train'], config['target_lang'], config, min_freq=2)

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

    train_data_loader = DataLoader(train_dataset, batch_size=config['train_batch_size'],
                                   shuffle=True, collate_fn=dataset_util.collate_fun)
    validation_data_loader = DataLoader(validation_dataset, batch_size=config['eval_batch_size'],
                                        collate_fn=dataset_util.collate_fun)
    test_data_loader = DataLoader(test_dataset, batch_size=config['eval_batch_size'],
                                  collate_fn=dataset_util.collate_fun)
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
    config = config_util.get_config('./config/config.yaml')
    preprocess(config)

if __name__ == '__main__':
    main()
