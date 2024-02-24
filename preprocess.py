import torch
from torch.utils.data import DataLoader

from datasets import load_dataset, Dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from dataset import BilingualDataset

import utils.dataset_util as dataset_util
from utils.config_util import get_config
import constants as const

from pathlib import Path

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
    dataset_dict = load_dataset(
        path=config['dataset_path'],
        name=config['dataset_name'],
    )
    num_rows = dataset_dict.num_rows

    # preprocessing
    print('Preprocessing sentences')
    dataset_dict = dataset_util.preprocess_sentences(dataset_dict)

    print('Building tokenizers from train dataset')
    src_tokenizer = tokenize(dataset_dict['train'], config['src_lang'], config, min_freq=2)
    target_tokenizer = tokenize(dataset_dict['train'], config['target_lang'], config, min_freq=2)

    print('Removing invalid sentences')
    num_reserved_tokens = 5
    dataset_dict = dataset_util.remove_invalid_sentences(dataset_dict,
                                                         src_tokenizer,
                                                         target_tokenizer,
                                                         config['seq_length'] - num_reserved_tokens)
    for dataset, num_row in dataset_dict.num_rows.items():
        if dataset in num_rows:
            print(f'Removed {num_rows[dataset] - num_row} sentences from {dataset}')

    train_dataset = BilingualDataset(
        dataset_dict['train'],
        src_tokenizer,
        target_tokenizer,
        config['src_lang'],
        config['target_lang'],
        config['seq_length']
    )
    validation_dataset = BilingualDataset(
        dataset_dict['validation'],
        src_tokenizer,
        target_tokenizer,
        config['src_lang'],
        config['target_lang'],
        config['seq_length']
    )
    test_dataset = BilingualDataset(
        dataset_dict['test'],
        src_tokenizer,
        target_tokenizer,
        config['src_lang'],
        config['target_lang'],
        config['seq_length']
    )

    train_data_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    validation_data_loader = DataLoader(validation_dataset, batch_size=1, shuffle=True)
    test_data_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    data_loaders = {
        'train': train_data_loader,
        'validation': validation_data_loader,
        'test': test_data_loader,
    }
    checkpoints_dir = Path(config['checkpoints_dir'])
    data_loaders_path = checkpoints_dir / config['data_loaders_basename']
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    torch.save(data_loaders, data_loaders_path)

if __name__ == '__main__':
    config = get_config('./config/config.yaml')
    preprocess(config)
