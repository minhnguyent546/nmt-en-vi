import html
import re
from typing import Any
from pathlib import Path

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from datasets import Dataset, DatasetDict
from tokenizers import Tokenizer

import underthesea
import contractions  # NOTE: this lib does not work well in some cases!

from nmt.billingual_dataset import BilingualDataset
from nmt.utils.misc import is_enabled
from nmt.constants import Config, SpecialToken


def make_data_loader(
    dataset: Dataset,
    src_tokenizer: Tokenizer,
    target_tokenizer: Tokenizer,
    *,
    batch_size: int,
    shuffle: bool = True,
    config: dict,
) -> DataLoader:
    bilingual_dataset = BilingualDataset(
        dataset,
        src_tokenizer,
        target_tokenizer,
        config['source'],
        config['target'],
        config['src_seq_length'],
    )

    assert src_tokenizer.token_to_id(SpecialToken.PAD) == target_tokenizer.token_to_id(SpecialToken.PAD)
    pad_token_id = src_tokenizer.token_to_id(SpecialToken.PAD)
    data_collator = CollatorWithPadding(
        pad_token_id,
        added_features=['encoder_input', 'decoder_input', 'labels']
    )
    data_loader = DataLoader(bilingual_dataset, batch_size=batch_size,
                             shuffle=shuffle, collate_fn=data_collator, pin_memory=True)

    return data_loader

def load_trained_tokenizers(config: dict) -> tuple[Tokenizer, Tokenizer]:
    checkpoints_dir = Path(config['checkpoints_dir'])
    if config['share_vocab']:
        src_tokenizer = Tokenizer.from_file(str(checkpoints_dir / config['tokenizer_basename'].format('combined')))
        target_tokenizer = src_tokenizer
    else:
        src_tokenizer = Tokenizer.from_file(str(checkpoints_dir / config['tokenizer_basename'].format(config['source'])))
        target_tokenizer = Tokenizer.from_file(str(checkpoints_dir / config['tokenizer_basename'].format(config['target'])))

    return src_tokenizer, target_tokenizer

def process_sentence(sentence: str, config: dict) -> str:
    # default actions
    sentence = sentence.strip()
    sentence = re.sub(r'\s(&[a-zA-Z]+;)', r'\1', sentence)
    sentence = html.unescape(sentence)

    if is_enabled(config, Config.LOWERCASE):
        sentence = sentence.lower()
    if is_enabled(config, Config.CONTRACTIONS):
        sentence = re.sub(r"\s('[\w\d]+)", r'\1', sentence)
        sentence = contractions.fix(sentence)
    if is_enabled(config, Config.VI_WORD_SEGMENTTATION):
        sentence = underthesea.word_tokenize(sentence, format='text')

    return sentence

def process_feature(dataset: DatasetDict, feature: str, config: dict) -> DatasetDict:
    dataset = dataset.map(lambda examples: {
        feature: [
            process_sentence(sentence, config)
            for sentence in examples[feature]
        ]
    }, batched=True)
    return dataset

def process_dataset_sentences(
    dataset: DatasetDict,
    config: dict,
) -> DatasetDict:
    for feature in ['source', 'target']:
        for split in dataset:
            if config[feature] not in dataset[split].features:
                break
        else:
            dataset = process_feature(dataset, config[feature], config['preprocess'][feature])

    return dataset

def _check_valid_pairs(
    examples: dict[str, list],
    src_tokenizer: Tokenizer,
    target_tokenizer: Tokenizer,
    max_seq_length: int,
    config: dict,
) -> list[bool]:
    if len(examples[config['source']]) != len(examples[config['target']]):
        raise ValueError('The number of source and target examples must be equal')

    valid_row = [True] * len(examples[config['source']])
    for row_id, (source, target) in enumerate(zip(examples[config['source']], examples[config['target']])):
        src_tokens_len = len(src_tokenizer.encode(source).ids)
        target_tokens_len = len(target_tokenizer.encode(target).ids)
        valid_row[row_id] = min(src_tokens_len, target_tokens_len) > 0 and \
                            max(src_tokens_len, target_tokens_len) <= max_seq_length

    return valid_row

def remove_invalid_pairs(
    dataset: DatasetDict,
    src_tokenizer: Tokenizer,
    target_tokenizer: Tokenizer,
    max_seq_length: int,
    config: dict,
) -> DatasetDict:
    return dataset.filter(lambda examples: _check_valid_pairs(
        examples,
        src_tokenizer,
        target_tokenizer,
        max_seq_length,
        config,
    ), batched=True)

class CollatorWithPadding:
    def __init__(self, padding_value: int, added_features: list[str] = []) -> None:
        self.padding_value = padding_value
        self.added_features = added_features

    def __call__(self, original_batch: list[dict[str, Any]]) -> dict[str, Any]:
        all_features = original_batch[0].keys()
        remain_features = [key for key in all_features if key not in self.added_features]

        feature_dict = {key: [item[key] for item in original_batch] for key in self.added_features}
        batch = {key: [item[key] for item in original_batch] for key in remain_features}
        feature_dict = {
            key: pad_sequence(value, batch_first=True, padding_value=self.padding_value)
            for key, value in feature_dict.items()
        }
        batch.update(feature_dict)
        return batch
