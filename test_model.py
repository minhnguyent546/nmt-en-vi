from pathlib import Path
import sys
import pandas as pd

import torch
import torch.nn as nn

from tokenizers import Tokenizer

import utils.model as model_util
import utils.config as config_util
import utils.bleu as bleu_util
import constants as const

def test_model(config):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    checkpoints_dir = Path(config['checkpoints_dir'])
    model_dir = checkpoints_dir / config['model_dir']
    model_dir.mkdir(parents=True, exist_ok=True)

    print('Loading data loaders')
    data_loaders = torch.load(checkpoints_dir / config['data_loaders_basename'])
    test_data_loader = data_loaders['test']

    print('Loading tokenizers')
    src_tokenizer = Tokenizer.from_file(str(checkpoints_dir / config['tokenizer_basename'].format(config['src_lang'])))
    target_tokenizer = Tokenizer.from_file(str(checkpoints_dir / config['tokenizer_basename'].format(config['target_lang'])))

    model = model_util.make_model(src_tokenizer, target_tokenizer, config)
    model.to(device)

    print('Loading latest model weights')
    model_latest_weights_path = model_util.get_latest_weights_file_path(config=config)
    if model_latest_weights_path is None:
        print('Aborted!')
        sys.exit(1)

    print(f'Loaded latest weights from: {model_latest_weights_path}')

    states = torch.load(model_latest_weights_path)

    model.load_state_dict(states['model_state_dict'])

    loss_function = nn.CrossEntropyLoss(ignore_index=src_tokenizer.token_to_id(const.PAD_TOKEN),
                                        label_smoothing=config['label_smoothing'])

    test_stats = model_util.evaluate(model, device, loss_function, test_data_loader,
                                     eval_max_steps=config['test_max_steps'])
    test_bleu = bleu_util.compute_dataset_bleu(model, device, test_data_loader.dataset,
                                              target_tokenizer, config['seq_length'],
                                              teacher_forcing=False,
                                              beam_size=config['beam_size'],
                                              beam_return_topk=config['beam_return_topk'],
                                              max_n=4, log_sentences=True,
                                              logging_interval=10)
    print(pd.DataFrame({
        'test_loss': [test_stats['eval_loss']],
        'test_accuracy': [test_stats['eval_accuracy']],
        'val_bleu-1': [test_bleu[0]],
        'val_bleu-2': [test_bleu[1]],
        'val_bleu-3': [test_bleu[2]],
        'val_bleu-4': [test_bleu[3]],
    }).to_string(index=False))

def main():
    config = config_util.get_config('./config/config.yaml')
    test_model(config)

if __name__ == '__main__':
    main()
