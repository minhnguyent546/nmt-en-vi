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

def preprocess_sentences(dataset):
    return dataset.map(lambda x: {
        'translation': {
            'en': x['translation']['en'].lower(),
            'vi': ViTokenizer.tokenize(x['translation']['vi']).lower(),
        }
    })

def remove_invalid_sentences(
    dataset,
    src_tokenizer: Tokenizer,
    target_tokenizer: Tokenizer,
    max_seq_length: int,
):
    result = []
    for item in dataset:
        src_tokens_len = len(src_tokenizer.encode(item['translation']['vi']).ids)
        target_tokens_len = len(target_tokenizer.encode(item['translation']['en']).ids)

        if min(src_tokens_len, target_tokens_len) > 0 and \
           max(src_tokens_len, target_tokens_len) <= max_seq_length:
           result.append(item)

    print(f'--> removed {len(dataset) - len(result)} invalid sentences')
    return result
