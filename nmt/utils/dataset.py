import html
import re
from typing import Any

from torch.nn.utils.rnn import pad_sequence

from datasets import DatasetDict
from tokenizers import Tokenizer

import underthesea
import contractions  # NOTE: this lib does not work well in some cases!
from nmt.utils.misc import is_enabled
from nmt.constants import Config

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
