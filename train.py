import torch
import torch.nn as nn
from torch import Tensor, device
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchtext.data.metrics import bleu_score

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from tqdm import tqdm # progress bar helper

from pathlib import Path
import warnings

from dataset import BilingualDataset
from config import get_config, get_weights_file_path

from transformer import Transformer, make_transformer

import utils.model_util as model_util
import utils.dataset_util as dataset_util

def tokenize_dataset(dataset, lang: str, config: dict, min_freq: int = 2) -> Tokenizer:
    tokenizer_dir = config['tokenizer_dir']
    tokenizer_basename = config['tokenizer_basename'].format(lang)
    tokenizer_path = Path(tokenizer_dir) / tokenizer_basename
    Path(tokenizer_dir).mkdir(parents=True, exist_ok=True)

    if Path.exists(tokenizer_path):
        return Tokenizer.from_file(str(tokenizer_path))

    tokenizer = Tokenizer(WordLevel(unk_token='<UNK>'))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = WordLevelTrainer(
        min_frequency=min_freq,
        special_tokens=['<UNK>', '<PAD>', '<SOS>', '<EOS>']
    )
    dataset_iter = dataset_util.create_iter_from_dataset(dataset, lang)
    tokenizer.train_from_iterator(dataset_iter, trainer=trainer)
    tokenizer.save(str(tokenizer_path))

    return tokenizer

def get_dataset(config: dict, split_rate: float = 0.9) -> tuple[DataLoader, DataLoader, Tokenizer, Tokenizer]:
    raw_dataset = load_dataset(
        path=config['dataset_path'],
        name=config['dataset_name'],
        split='train',
    )

    # preprocessing
    print('preprocessing sentences')
    raw_dataset = dataset_util.preprocess_sentences(raw_dataset)

    print(f'spliting dataset with split rate = {split_rate}')
    raw_train_dataset, raw_validation_dataset = dataset_util.split_dataset(raw_dataset, split_rate=split_rate)

    print('building tokenizers from train dataset')
    src_tokenizer = tokenize_dataset(raw_train_dataset, config['src_lang'], config)
    target_tokenizer = tokenize_dataset(raw_train_dataset, config['target_lang'], config)

    print('removing invalid sentences from train dataset')
    num_reserved_tokens = 5
    raw_train_dataset = dataset_util.remove_invalid_sentences(
        raw_train_dataset,
        src_tokenizer,
        target_tokenizer,
        config['seq_length'] - num_reserved_tokens
    )

    print('removing invalid sentences from validation dataset')
    raw_validation_dataset = dataset_util.remove_invalid_sentences(
        raw_validation_dataset,
        src_tokenizer,
        target_tokenizer,
        config['seq_length'] - num_reserved_tokens
    )

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
    for item in raw_train_dataset + raw_validation_dataset:
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

def greedy_search_decode(
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
        decoder_mask = model_util.create_mask(decoder_input.size(1)).type_as(src_mask).to(device)
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

def evaluate_model(
    model: Transformer,
    device: device,
    validation_data_loader: DataLoader,
    src_tokenizer: Tokenizer,
    target_tokenizer: Tokenizer,
    seq_length: int,
    global_step: int,
    print_message,
    writer: SummaryWriter | None = None,
    num_samples: int = -1,
):
    model.eval()
    counter = 0

    src_texts = []
    target_texts = []
    predicted_texts = []

    with torch.no_grad():
        # environment with no gradient calculation
        for batch in validation_data_loader:
            encoder_input = batch['encoder_input'].to(device) # (batch_size, seq_length)
            encoder_mask = batch['encoder_mask'].to(device) # (batch_size, 1, 1, seq_length)
            label = batch['label'].to(device)

            batch_size = encoder_input.size(0)
            assert batch_size == 1, 'batch_size must be 1 for validation'

            model_output = greedy_search_decode(
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

            src_text_tokens = src_tokenizer.encode(src_text).tokens
            target_text_tokens = target_tokenizer.encode(target_text).tokens
            predicted_text_tokens = target_tokenizer.encode(predicted_text).tokens

            src_texts.append([token for token in src_text_tokens if token != '<UNK>'])
            target_texts.append([[token for token in target_text_tokens if token != '<UNK>']])
            predicted_texts.append([token for token in predicted_text_tokens if token != '<UNK>'])

            print_message('-'*80)
            print(f'[{counter + 1}/{num_samples}] source   : {src_text}')
            print(f'[{counter + 1}/{num_samples}] target   : {target_text}')
            print(f'[{counter + 1}/{num_samples}] predicted: {predicted_text}')

            counter += 1
            if num_samples > 0 and counter == num_samples:
                break

    if writer is not None:
        _bleu_score = bleu_score(predicted_texts, target_texts, max_n=4)
        writer.add_scalar('evaluation bleu score', _bleu_score, global_step=global_step)
        writer.flush()
        print('>> evaluation bleu score:', _bleu_score)

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

    model = make_model(src_vocab_size, target_vocab_size, config)
    model.to(device)

    # Tensorboard
    writer = SummaryWriter(config['experiment_name'])

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

    loss_function = nn.CrossEntropyLoss(ignore_index=src_tokenizer.token_to_id('<PAD>'), label_smoothing=0.1)

    num_epochs = config['num_epochs']
    log_step = config['log_step']
    running_loss = 0.0
    for epoch in range(initial_epoch, num_epochs):
        # clear cuda cache
        torch.cuda.empty_cache()

        model.train()

        epoch_loss = 0.0
        batch_iterator = tqdm(train_data_loader, desc=f'processing epoch {epoch + 1:02d}/{num_epochs:02d}')
        batch_message_printer = lambda message: batch_iterator.write(message)
        for batch_idx, batch in enumerate(batch_iterator):
            encoder_input = batch['encoder_input'].to(device) # (batch_size, seq_length)
            decoder_input = batch['decoder_input'].to(device) # (batch_size, seq_length)
            encoder_mask = batch['encoder_mask'].to(device) # (batch_size, 1, 1, seq_length)
            decoder_mask = batch['decoder_mask'].to(device) # (batch_size, 1, seq_length, seq_length)

            encoder_output = model.encode(encoder_input, encoder_mask) # (batch_size, seq_length, d_model)
            decoder_output = model.decode(encoder_output, decoder_input, encoder_mask, decoder_mask) # (batch_size, seq_length, d_model)
            projected_output = model.project(decoder_output) # (batch_size, seq_length, target_vocab_size)
            labels = batch['label'].to(device) # (batch_size, seq_length)

            # calculate the loss
            # projected_output: (batch_size * seq_length, target_vocab_size)
            # label: (batch_size * seq_length)
            loss = loss_function(projected_output.view(-1, target_tokenizer.get_vocab_size()), labels.view(-1))

            batch_iterator.set_postfix({'loss': f'{loss.item():0.3f}'})
            # log the loss
            writer.add_scalar('loss/batch_loss', loss.item(), global_step)
            writer.flush()

            # backpropagte the loss
            loss.backward()

            # update the weights
            optimizer.step()
            optimizer.zero_grad()

            running_loss += loss.item()

            if isinstance(log_step, int) and (batch_idx + 1) % log_step == 0:
                # run validation
                evaluate_model(
                    model,
                    device,
                    validation_data_loader,
                    src_tokenizer,
                    target_tokenizer,
                    config['seq_length'],
                    global_step,
                    batch_message_printer,
                    writer=writer,
                    num_samples=config['num_eval_samples'],
                )
                writer.add_scalar('loss/running_loss', running_loss / log_step, global_step)
                writer.flush()
                running_loss = 0.0

            global_step += 1

        if isinstance(log_step, str) and log_step.lower() == 'epoch':
            evaluate_model(
                model,
                device,
                validation_data_loader,
                src_tokenizer,
                target_tokenizer,
                config['seq_length'],
                global_step,
                batch_message_printer,
                writer=writer,
                num_samples=config['num_eval_samples'],
            )
            writer.add_scalar('loss/epoch_loss', epoch_loss / len(batch_iterator), global_step)
            writer.flush()

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
