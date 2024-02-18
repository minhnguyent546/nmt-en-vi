import torch

from tokenizers import Tokenizer

from pathlib import Path

from config import get_config
import utils.model_util as model_util

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

    src_vocab_size, target_vocab_size = src_tokenizer.get_vocab_size(), target_tokenizer.get_vocab_size()
    model = model_util.make_model(src_vocab_size, target_vocab_size, config)
    model.to(device)

    print('Loading latest model weights')
    model_filename = model_util.get_latest_weights_file_path(config=config)
    if model_filename is None:
        print('Aborted!')
        exit(1)

    states = torch.load(model_filename, map_location=torch.device('cpu'))

    model.load_state_dict(states['model_state_dict'])

    model_util.evaluate_model(
        model,
        device,
        test_data_loader,
        src_tokenizer,
        target_tokenizer,
        config['seq_length'],
        print,
        beam_size=config['beam_size'],
        num_samples=config['num_test_samples'],
    )

if __name__ == '__main__':
    config = get_config()
    test_model(config)
