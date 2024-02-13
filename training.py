import torch
import torch.nn as nn
from torch import Tensor, device
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from datasets import load_dataset, Dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from pyvi import ViTokenizer # NLP lib for vietnamese

from tqdm import tqdm # progress bar helper

from pathlib import Path
import warnings

from dataset import BilingualDataset
from config import get_config, get_weights_file_path

from transformer import Transformer, make_transformer
from utils import create_mask

def create_iter_from_dataset(dataset, lang: str):
    for item in dataset:
        yield item['translation'][lang]

def get_tokenzier(dataset, lang: str, config: dict) -> Tokenizer:
    tokenizer_dir = config['tokenizer_dir']
    tokenizer_basename = config['tokenizer_basename'].format(lang)
    tokenizer_path = Path(tokenizer_dir) / tokenizer_basename
    Path(tokenizer_dir).mkdir(parents=True, exist_ok=True)

    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token='<UNK>'))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(
            min_frequency=2,
            special_tokens=['<UNK>', '<PAD>', '<SOS>', '<EOS>']
        )
        dataset_iter = create_iter_from_dataset(dataset, lang)
        tokenizer.train_from_iterator(dataset_iter, trainer=trainer)
        tokenizer.enable_truncation(max_length=config['seq_length'] - 5) # reserve some space for special tokens
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

    return tokenizer

def get_dataset(config: dict) -> tuple[DataLoader, DataLoader, Tokenizer, Tokenizer]:
    raw_dataset = load_dataset(
        path=config['dataset_path'],
        name=config['dataset_name'],
        split='train',
    )

    # preprocessing
    print('preprocessing dataset')
    raw_dataset = raw_dataset.map(lambda x: {
        'translation': {
            'vi': ViTokenizer.tokenize(x['translation']['vi']),
            'en': x['translation']['en']
        }
    })

    print('building tokenizers')
    src_tokenizer = get_tokenzier(raw_dataset, config['src_lang'], config)
    target_tokenizer = get_tokenzier(raw_dataset, config['target_lang'], config)

    print('spliting dataset')
    train_dataset_size = int(len(raw_dataset) * 0.9)
    validation_dataset_size = len(raw_dataset) - train_dataset_size
    raw_train_dataset, raw_validation_dataset = random_split(raw_dataset, [train_dataset_size, validation_dataset_size])

    train_dataset = BilingualDataset(
        raw_train_dataset,
        src_tokenizer,
        target_tokenizer,
        config['src_lang'],
        config['target_lang'],
        config['seq_length']
    )
    validation_dataset = BilingualDataset(
        raw_validation_dataset,
        src_tokenizer,
        target_tokenizer,
        config['src_lang'],
        config['target_lang'],
        config['seq_length']
    )

    max_src_seq_length = 0
    max_target_seq_length = 0
    for item in raw_dataset:    
        src_seq_length = len(src_tokenizer.encode(item['translation'][config['src_lang']]).ids)
        target_seq_length = len(target_tokenizer.encode(item['translation'][config['target_lang']]).ids)
        max_src_seq_length = max(max_src_seq_length, src_seq_length)
        max_target_seq_length = max(max_target_seq_length, target_seq_length)

    print('max_src_seq_legnth:', max_src_seq_length)
    print('max_target_seq_length:', max_target_seq_length)
    print('using seq_length:', config['seq_length'])

    train_data_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    validation_data_loader = DataLoader(validation_dataset, batch_size=1, shuffle=True)

    return train_data_loader, validation_data_loader, src_tokenizer, target_tokenizer

def greedy_decode(
    model: Transformer,
    device: device,
    src: Tensor,
    src_mask: Tensor,
    src_tokenizer: Tokenizer,
    target_tokenizer: Tokenizer,
    seq_length: int
):
    sos_token_id = target_tokenizer.token_to_id('<SOS>')
    eos_token_id = target_tokenizer.token_to_id('<EOS>')

    encoder_output = model.encode(src, src_mask) # (batch_size, seq_length, d_model)

    # initialize decoder input (sos token)
    decoder_input = torch.empty((1, 1)).fill_(sos_token_id).type_as(src).to(device)
    for i in range(seq_length):
        decoder_mask = create_mask(decoder_input.size(1)).type_as(src_mask).to(device)
        decoder_output = model.decode(encoder_output, decoder_input, src_mask, decoder_mask)

        # get the next token
        projected_output = model.project(decoder_output[:, -1])
        next_token = torch.argmax(projected_output, dim=1)

        decoder_input = torch.cat([
            decoder_input,
            torch.empty((1, 1)).type_as(src).fill_(next_token.item()).to(device)
        ], dim=1)

        if next_token == eos_token_id:
            break

    return decoder_input.squeeze(0)

def run_validation(
    model: Transformer,
    device: device,
    validation_data_loader: DataLoader,
    src_tokenizer: Tokenizer,
    target_tokenizer: Tokenizer,
    seq_length: int,
    print_message,
    summary_writer: SummaryWriter | None = None,
    num_examples: int = 5,
):
    model.eval()
    counter = 0

    src_texts = []
    target_texts = []
    predicted_texts = []

    with torch.no_grad():
        # environment with no gradient calculation
        for batch in validation_data_loader:
            counter += 1
            encoder_input = batch['encoder_input'].to(device) # (batch_size, seq_length)
            encoder_mask = batch['encoder_mask'].to(device) # (batch_size, 1, 1, seq_length)

            batch_size = encoder_input.size(0)
            assert batch_size == 1, 'batch_size must be 1 for validation'

            model_output = greedy_decode(
                model,
                device,
                encoder_input,
                encoder_mask,
                src_tokenizer,
                target_tokenizer,
                seq_length
            )
            src_text = batch['src_text'][0]
            target_text = batch['target_text'][0]
            predicted_text = target_tokenizer.decode(model_output.detach().cpu().numpy())

            src_texts.append(src_text)
            target_texts.append(target_text)
            predicted_texts.append(predicted_text)

            print_message('-'*80)
            print(f'[{counter}/{num_examples}] source   : {src_text}')
            print(f'[{counter}/{num_examples}] target   : {target_text}')
            print(f'[{counter}/{num_examples}] predicted: {predicted_text}')

            if counter == num_examples:
                break

    if summary_writer is not None:
        # calculate BLEU score
        pass

def make_model(src_vocab_size: int, target_vocab_size: int, config: dict) -> Transformer:
    model = make_transformer(
        src_vocab_size,
        target_vocab_size,
        config['seq_length'],
        config['seq_length'],
        d_model=config['d_model'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        d_ffn=config['d_ffn'],
        dropout_rate=config['dropout_rate'],
    )
    return model

def train_model(config):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'using device {device}')
    device = torch.device(device)

    Path(config['model_dir']).mkdir(parents=True, exist_ok=True)

    train_data_loader, validation_data_loader, src_tokenizer, target_tokenizer = get_dataset(config)
    src_vocab_size = src_tokenizer.get_vocab_size()
    target_vocab_size = target_tokenizer.get_vocab_size()
    model = make_model(src_vocab_size, target_vocab_size, config).to(device)

    # Tensorboard
    summary_writer = SummaryWriter(config['experiment_name'])

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

    initial_epoch = 0
    global_step = 0
    if config['preload'] is not None:
        model_filename = get_weights_file_path(epoch=config['preload'], config=config)
        print(f'loading weights from {model_filename}')
        states = torch.load(model_filename)

        # continue from previous completed epoch
        initial_epoch = states['epoch'] + 1
        
        model.load_state_dict(states['model_state_dict'])
        optimizer.load_state_dict(states['optimizer_state_dict'])
        global_step = states['global_step']

    loss_function = nn.CrossEntropyLoss(ignore_index=src_tokenizer.token_to_id('<PAD>'), label_smoothing=0.1).to(device)

    num_epochs = config['num_epochs']
    for epoch in range(initial_epoch, num_epochs):
        # clear cuda cache
        torch.cuda.empty_cache()

        model.train()

        batch_iterator = tqdm(train_data_loader, desc=f'processing epoch {epoch:02d}/{num_epochs - 1}')
        batch_message_printer = lambda message: batch_iterator.write(message)
        iter_counter = 0
        for batch in batch_iterator:
            encoder_input = batch['encoder_input'].to(device) # (batch_size, seq_length)
            decoder_input = batch['decoder_input'].to(device) # (batch_size, seq_length)
            encoder_mask = batch['encoder_mask'].to(device) # (batch_size, 1, 1, seq_length)
            decoder_mask = batch['decoder_mask'].to(device) # (batch_size, 1, seq_length, seq_length)

            encoder_output = model.encode(encoder_input, encoder_mask) # (batch_size, seq_length, d_model)
            decoder_output = model.decode(encoder_output, decoder_input, encoder_mask, decoder_mask) # (batch_size, seq_length, d_model)
            projected_output = model.project(decoder_output) # (batch_size, seq_length, target_vocab_size)
            label = batch['label'].to(device) # (batch_size, seq_length)

            # calculate the loss
            # projected_output: (batch_size * seq_length, target_vocab_size)
            # label: (batch_size * seq_length)
            loss = loss_function(projected_output.view(-1, target_tokenizer.get_vocab_size()), label.view(-1))

            batch_iterator.set_postfix({'loss': f'{loss.item():0.3f}'})
            # log the loss
            summary_writer.add_scalar('loss', loss.item(), global_step)
            summary_writer.flush()

            # backpropagte the loss
            loss.backward()

            # update the weights
            optimizer.step()
            optimizer.zero_grad()

            if iter_counter % 10 == 0:
                # run validation
                run_validation(
                    model,
                    device,
                    validation_data_loader,
                    src_tokenizer,
                    target_tokenizer,
                    config['seq_length'],
                    batch_message_printer
                )

            global_step += 1
            iter_counter += 1

        # save the model after every epoch
        model_filename = get_weights_file_path(f'{epoch:02d}', config)
        torch.save({
            'epoch': epoch,
            'global_step': global_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, model_filename)

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    config = get_config()
    train_model(config)
