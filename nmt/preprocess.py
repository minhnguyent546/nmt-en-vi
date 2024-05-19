import argparse
import pandas as pd
from pathlib import Path
from typing import Generator, Literal

from datasets import DatasetDict, load_dataset
import tokenizers
from tokenizers import Tokenizer
import tokenizers.decoders
import tokenizers.models
from tokenizers.pre_tokenizers import Whitespace
import tokenizers.trainers

from nmt.constants import SpecialToken, TokenizerModel
from nmt.utils import (
    config as config_util,
    dataset as dataset_util,
    misc as misc_util,
)
from nmt.utils.logging import init_logger, logger
from nmt.utils.misc import set_seed


def get_tokenizer_trainer(
    tokenizer_model: str,
    *,
    vocab_size: int = 30_000,
    min_freq: int | Literal['default'] = 'default',
    show_progress: bool = True,
) -> tuple[Tokenizer, tokenizers.trainers.Trainer]:
    tokenizer_model = tokenizer_model.lower()

    if min_freq == 'default':
        min_freq_mapping = {
            TokenizerModel.WORD_LEVEL: 2,
            TokenizerModel.BPE: 1,
            TokenizerModel.WORD_PIECE: 1,
        }
        min_freq = min_freq_mapping.get(tokenizer_model, 1)
    all_special_tokens = [SpecialToken.PAD, SpecialToken.SOS, SpecialToken.EOS, SpecialToken.UNK]
    if tokenizer_model == TokenizerModel.WORD_LEVEL:
        tokenizer = Tokenizer(tokenizers.models.WordLevel(unk_token=SpecialToken.UNK))  # pyright: ignore[reportCallIssue]
        tokenizer.pre_tokenizer = Whitespace()

        trainer = tokenizers.trainers.WordLevelTrainer(
            vocab_size=vocab_size,
            min_frequency=min_freq,
            show_progress=show_progress,
            special_tokens=all_special_tokens,
        )
    elif tokenizer_model == TokenizerModel.BPE:
        tokenizer = Tokenizer(tokenizers.models.BPE(unk_token=SpecialToken.UNK))
        tokenizer.pre_tokenizer = Whitespace()
        tokenizer.decoder = tokenizers.decoders.BPEDecoder(suffix=SpecialToken.BPE_SUFFIX)

        trainer = tokenizers.trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_freq,
            show_progress=show_progress,
            special_tokens=all_special_tokens,
            end_of_word_suffix=SpecialToken.BPE_SUFFIX,
        )
    elif tokenizer_model == TokenizerModel.WORD_PIECE:
        tokenizer = Tokenizer(tokenizers.models.WordPiece(
            unk_token=SpecialToken.UNK,
            max_input_chars_per_word=100,
        ))  # pyright: ignore[reportCallIssue]
        tokenizer.pre_tokenizer = Whitespace()
        tokenizer.decoder = tokenizers.decoders.WordPiece(
            prefix=SpecialToken.WORD_PIECE_PREFIX,
            cleanup=False,
        )

        trainer = tokenizers.trainers.WordPieceTrainer(
            vocab_size=vocab_size,
            min_frequency=min_freq,
            show_progress=show_progress,
            special_tokens=all_special_tokens,
            continuing_subword_prefix=SpecialToken.WORD_PIECE_PREFIX,
        )
    else:
        raise ValueError(f'Unsupported tokenizer model "{tokenizer_model}". Possible values are word_level, bpe, and word_piece')

    return tokenizer, trainer

def tokenize(
    data_iter: Generator[str, None, None],
    feature: str,
    config: dict,
    *,
    vocab_size: int = 30_000,
    min_freq: int | Literal['default'] = 'default',
) -> Tokenizer:
    checkpoints_dir = Path(config['checkpoints_dir'])
    tokenizer_path = checkpoints_dir / config['tokenizer_basename'].format(feature)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    tokenizer, trainer = get_tokenizer_trainer(
        config['tokenizer_model'],
        vocab_size=vocab_size,
        min_freq=min_freq,
    )

    tokenizer.train_from_iterator(data_iter, trainer=trainer)
    tokenizer.save(str(tokenizer_path))

    return tokenizer

def _load_datasets(config: dict) -> DatasetDict:
    """
    Each dataset in the dataset dict should have two features: ``config['source']`` and ``config['target']``,
    """
    raw_datasets: DatasetDict = load_dataset(
        path=config['dataset_path'],
        name=config['dataset_name'],
        data_files=config['data_files'],
        **config['dataset_config_kwags'],
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
    init_logger()

    raw_datasets = _load_datasets(config)
    num_rows = raw_datasets.num_rows
    raw_datasets = dataset_util.process_dataset_sentences(raw_datasets, config)

    for split_name in ['train', 'validation', 'test']:
        if split_name in raw_datasets:
            logger.info(pd.DataFrame(raw_datasets[split_name][:5]))

    logger.info('Building tokenizers from train dataset')
    if config['share_vocab']:
        data_iter = misc_util.combined_iterator(
            raw_datasets['train'][config['source']],
            raw_datasets['train'][config['target']],
        )
        src_tokenizer = tokenize(data_iter, 'combined', config, vocab_size=config['source_vocab_size'])
        target_tokenizer = src_tokenizer
        logger.info('Size of vocabulary: %d', src_tokenizer.get_vocab_size())
    else:
        src_data_iter = (item for item in raw_datasets['train'][config['source']])
        target_data_iter = (item for item in raw_datasets['train'][config['target']])
        src_tokenizer = tokenize(src_data_iter, config['source'], config, vocab_size=config['source_vocab_size'])
        target_tokenizer = tokenize(target_data_iter, config['target'], config, vocab_size=config['target_vocab_size'])
        logger.info('Size of src vocabulary: %d', src_tokenizer.get_vocab_size())
        logger.info('Size of target vocabulary: %d', target_tokenizer.get_vocab_size())

    logger.info('Removing invalid pairs')
    num_reserved_tokens = 2  # for SOS and EOS tokens
    raw_datasets = dataset_util.remove_invalid_pairs(raw_datasets,
                                                     src_tokenizer,
                                                     target_tokenizer,
                                                     config['src_seq_length'] - num_reserved_tokens,
                                                     config)

    for dataset, num_row in raw_datasets.num_rows.items():
        if dataset in num_rows:
            logger.info('Removed %d pairs from %s', num_rows[dataset] - num_row, dataset)

    raw_datasets.save_to_disk(config['dataset_save_path'])
    logger.info('Saved dataset to %s', config["dataset_save_path"])

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
