from datasets import Dataset, DatasetDict
from torch.utils.data import random_split

from pyvi import ViTokenizer

from tokenizers import Tokenizer

def create_iter_from_dataset(dataset, lang: str):
    for item in dataset:
        yield item['translation'][lang]

def split_dataset(dataset, split_rate: float = 0.9):
    dataset_size = len(dataset)
    train_dataset_size = int(dataset_size * split_rate)
    validation_dataset_size = dataset_size - train_dataset_size

    train_dataset, validation_dataset = random_split(dataset, [train_dataset_size, validation_dataset_size])
    return train_dataset, validation_dataset

def preprocess_sentences(dataset: Dataset | DatasetDict) -> Dataset | DatasetDict:
    return dataset.map(lambda item: {
        'translation': {
            'en': item['translation']['en'].lower(),
            'vi': ViTokenizer.tokenize(item['translation']['vi']).lower(),
        }
    })

def is_valid_pair(
    item,
    src_tokenizer: Tokenizer,
    target_tokenizer: Tokenizer,
    max_seq_length: int,
) -> bool:
    src_tokens_len = len(src_tokenizer.encode(item['translation']['en']).ids)
    target_tokens_len = len(target_tokenizer.encode(item['translation']['vi']).ids)

    return min(src_tokens_len, target_tokens_len) > 0 and \
           max(src_tokens_len, target_tokens_len) <= max_seq_length

def remove_invalid_sentences(
    dataset: Dataset | DatasetDict,
    src_tokenizer: Tokenizer,
    target_tokenizer: Tokenizer,
    max_seq_length: int,
) -> Dataset | DatasetDict:
    return dataset.filter(lambda item: is_valid_pair(item,
                                                     src_tokenizer,
                                                     target_tokenizer,
                                                     max_seq_length))
