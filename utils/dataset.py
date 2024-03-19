import html
import re

from torch.utils.data import random_split
from torch.nn.utils.rnn import pad_sequence

from datasets import Dataset, DatasetDict
from tokenizers import Tokenizer

import underthesea
import contractions  # NOTE: this lib is not work well in some cases!

def create_iter_from_dataset(dataset, lang: str):
    for item in dataset:
        yield item['translation'][lang]

def split_dataset(dataset, split_rate: float = 0.9):
    dataset_size = len(dataset)
    train_dataset_size = int(dataset_size * split_rate)
    validation_dataset_size = dataset_size - train_dataset_size

    train_dataset, validation_dataset = random_split(dataset, [train_dataset_size, validation_dataset_size])
    return train_dataset, validation_dataset

def collate_fun(original_batch):
    all_keys = original_batch[0].keys()
    added_keys = ['encoder_input', 'decoder_input', 'labels']
    remain_keys = [key for key in all_keys if key not in added_keys]

    groups = {key: [item[key] for item in original_batch] for key in added_keys}
    batch = {key: [item[key] for item in original_batch] for key in remain_keys}
    groups = {key: pad_sequence(group, batch_first=True) for key, group in groups.items()}
    batch.update(groups)
    return batch

def _process_en_sentence(sentence: str) -> str:
    sentence = sentence.strip().lower()
    sentence = re.sub(r" (&[a-zA-Z]+;)", r"\1", sentence)
    sentence = html.unescape(sentence)
    sentence = contractions.fix(sentence)
    return sentence

def _process_vi_sentence(sentence: str, vi_config: dict) -> str:
    sentence = sentence.strip().lower()
    sentence = html.unescape(sentence)
    if 'vi_word_segmentation' in vi_config and vi_config['vi_word_segmentation']:
        sentence = underthesea.word_tokenize(sentence, format='text')
    return sentence

def _process_en_sentences(examples):
    if isinstance(examples['translation'], list):
        # when batched=True
        return {
            'translation': [
                {k: v if k != 'en' else _process_en_sentence(v) for k, v in item.items()}
                for item in examples['translation']
            ]
        }
    else:
        return {
            'translation': {k: v if k != 'en' else _process_en_sentence(v) for k, v in examples['translation'].items()}
        }

def _process_vi_sentences(examples, vi_config: dict):
    if isinstance(examples['translation'], list):
        # when batched=True
        return {
            'translation': [
                {k: v if k != 'vi' else _process_vi_sentence(v, vi_config) for k, v in item.items()}
                for item in examples['translation']
            ]
        }
    else:
        return {
            'translation': {k: v if k != 'vi' else _process_vi_sentence(v, vi_config) for k, v in examples['translation'].items()}
        }

def _process_sentences_by_lang(
    dataset: Dataset | DatasetDict,
    lang: str,
    vi_config: dict,
    **kwargs,
) -> Dataset | DatasetDict:
    if lang == 'vi':
        return dataset.map(lambda item: _process_vi_sentences(item, vi_config), **kwargs)
    elif lang == 'en':
        return dataset.map(_process_en_sentences, **kwargs)
    else:
        raise ValueError(f'Unsupported language: {lang}')

def process_dataset_sentences(
    dataset: Dataset | DatasetDict,
    langs: str | list[str],
    vi_config: dict = {},
    **kwargs,
) -> Dataset | DatasetDict:
    if isinstance(langs, str):
        langs = [langs]

    for lang in langs:
        print(f'Processing {lang} sentences')
        dataset = _process_sentences_by_lang(dataset, lang, vi_config, **kwargs)

    return dataset

def _check_valid_pairs(
    examples,
    src_tokenizer: Tokenizer,
    target_tokenizer: Tokenizer,
    max_seq_length: int,
    src_lang: str,
    target_lang: str,
) -> bool | list[bool]:
    if isinstance(examples['translation'], list):
        is_valid = [True] * len(examples['translation'])

        for i, item in enumerate(examples['translation']):
            src_tokens_len = len(src_tokenizer.encode(item[src_lang]).ids)
            target_tokens_len = len(target_tokenizer.encode(item[target_lang]).ids)

            is_valid[i] = min(src_tokens_len, target_tokens_len) > 0 and \
                          max(src_tokens_len, target_tokens_len) <= max_seq_length
        return is_valid
    else:
        src_tokens_len = len(src_tokenizer.encode(examples['translation'][src_lang]).ids)
        target_tokens_len = len(target_tokenizer.encode(examples['translation'][target_lang]).ids)

        return min(src_tokens_len, target_tokens_len) > 0 and \
               max(src_tokens_len, target_tokens_len) <= max_seq_length

def remove_invalid_sentences(
    dataset: Dataset | DatasetDict,
    src_tokenizer: Tokenizer,
    target_tokenizer: Tokenizer,
    max_seq_length: int,
    src_lang: str,
    target_lang: str,
    **kwargs,
) -> Dataset | DatasetDict:
    return dataset.filter(
        lambda examples: _check_valid_pairs(
            examples,
            src_tokenizer,
            target_tokenizer,
            max_seq_length,
            src_lang,
            target_lang,
        ),
        **kwargs,
    )
