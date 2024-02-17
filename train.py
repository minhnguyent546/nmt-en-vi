import torch
import torch.nn as nn
import torch.nn.functional as Fun
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
from config import get_config

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
    target_tokenizer: Tokenizer,
    seq_length: int
) -> Tensor:
    # assume batch_size is 1
    sos_token_id = target_tokenizer.token_to_id('<SOS>')
    eos_token_id = target_tokenizer.token_to_id('<EOS>')

    encoder_output = model.encode(src, src_mask) # (batch_size, seq_length, d_model)

    # initialize decoder input that contains only <SOS> token
    decoder_input = torch.empty((1, 1)).fill_(sos_token_id).type_as(src).to(device)
    for i in range(seq_length):
        # create mask for decoder input
        decoder_mask = model_util.create_mask(decoder_input.size(1)).type_as(src_mask).to(device)

        # decode
        decoder_output = model.decode(encoder_output, decoder_input, src_mask, decoder_mask)

        # get token with highest probability
        projected_output = model.project(decoder_output[:, -1, :])
        next_token = torch.argmax(projected_output, dim=1)

        # concatenate the next token to the decoder input for the next prediction
        decoder_input = torch.cat([
            decoder_input,
            torch.empty((1, 1)).type_as(src).fill_(next_token.item()).to(device)
        ], dim=1)

        # if we reach the <EOS> token, then stop
        if next_token == eos_token_id:
            break

    return decoder_input.squeeze(0)

def length_penalty(length: int, alpha: float = 0.6) -> float:
    """
    as formula described in We at al. (2016)
    """
    return (5 + length) ** alpha / (5 + 1) ** alpha

def beam_search_decode(
    model: Transformer,
    device: device,
    beam_size: int,
    src: Tensor,
    src_mask: Tensor,
    target_tokenizer: Tokenizer,
    seq_length: int,
) -> Tensor:
    # assume batch_size is 1
    sos_token_id = target_tokenizer.token_to_id('<SOS>')
    eos_token_id = target_tokenizer.token_to_id('<EOS>')

    encoder_output = model.encode(src, src_mask) # (batch_size, seq_length, d_model)

    # initialize decoder input that contains only <SOS> token
    decoder_input = torch.empty((1, 1)).fill_(sos_token_id).type_as(src).to(device)

    # candidate list of tuple (decoder_input, log_score)
    cands = [(decoder_input, 0.0)]
    for i in range(seq_length):
        new_cands = []

        for cand, log_score in cands:
            # do not expand the candidate that have reached <EOS> token
            if cand[0, -1].item() == eos_token_id:
                continue

            # create mask for decoder input
            cand_mask = model_util.create_mask(cand.size(1)).type_as(src_mask).to(device)

            # decode
            output = model.decode(encoder_output, cand, src_mask, cand_mask)

            # get next token probabilities
            # projected_output: shape ``(1, target_vocab_size)``
            # topk_prob       : shape ``(1, beam_size)``
            # topk_token      : shape ``(1, beam_size)``
            projected_output = model.project(output[:, -1, :])
            projected_output = Fun.log_softmax(projected_output, dim=-1) / length_penalty(cand.size(1) + 1)
            # get the top k largest tokens
            topk_token_prob, topk_token = torch.topk(projected_output, beam_size, dim=1)
            for j in range(beam_size):
                # token: shape ``(1, 1)``
                # token_prob: scalar
                token = topk_token[0][j].unsqueeze(0).unsqueeze(0)
                token_prob = topk_token_prob[0][j].item()

                new_cand = torch.cat([
                    cand,
                    token
                ], dim=1)

                new_cands.append((new_cand, log_score + token_prob))

        cands = sorted(new_cands, key=lambda x: x[1], reverse=True)
        cands = cands[:beam_size]

        if all([cand[0][-1].item() == eos_token_id for cand, _ in cands]):
            break

    return cands[0][0].squeeze(0)

def evaluate_model(
    model: Transformer,
    device: device,
    validation_data_loader: DataLoader,
    src_tokenizer: Tokenizer,
    target_tokenizer: Tokenizer,
    seq_length: int,
    epoch: int,
    print_message,
    beam_size: int | None = None,
    writer: SummaryWriter | None = None,
    num_samples: int = -1,
) -> None:
    model.eval()
    counter = 0

    src_texts = []
    target_texts = []
    predicted_texts = []
    if num_samples == -1:
        num_samples = len(validation_data_loader)

    with torch.no_grad():
        # environment with no gradient calculation
        ignored_tokens = ['<UNK>']
        for batch in validation_data_loader:
            encoder_input = batch['encoder_input'].to(device) # (batch_size, seq_length)
            encoder_mask = batch['encoder_mask'].to(device) # (batch_size, 1, 1, seq_length)
            label = batch['label'].to(device)

            batch_size = encoder_input.size(0)
            assert batch_size == 1, 'batch_size must be 1 for validation'

            if beam_size is not None:
                model_output = beam_search_decode(model, device, beam_size, encoder_input,
                                                  encoder_mask, target_tokenizer, seq_length)
            else:
                model_output = greedy_search_decode(model, device, encoder_input,
                                                    encoder_mask, target_tokenizer, seq_length)

            src_text = batch['src_text'][0]
            target_text = batch['target_text'][0]
            predicted_text = target_tokenizer.decode(model_output.detach().cpu().numpy())

            src_text_tokens = [token for token in src_tokenizer.encode(src_text).tokens if token not in ignored_tokens]
            target_text_tokens = [token for token in target_tokenizer.encode(target_text).tokens if token not in ignored_tokens]
            predicted_text_tokens = [token for token in target_tokenizer.encode(predicted_text).tokens if token not in ignored_tokens]

            src_texts.append(src_text_tokens)
            target_texts.append([target_text_tokens])
            predicted_texts.append(predicted_text_tokens)

            batch_bleu_scores = []
            for n_gram in range(1, 5):
                score = bleu_score(
                    [predicted_text_tokens],
                    [[target_text_tokens]],
                    max_n=n_gram,
                    weights=[1 / n_gram] * n_gram
                )
                batch_bleu_scores.append(score)

            print_message('-'*80)
            print_message(f'[{counter + 1}/{num_samples}] source    : {src_text}')
            print_message(f'[{counter + 1}/{num_samples}] target    : {target_text}')
            print_message(f'[{counter + 1}/{num_samples}] predicted : {predicted_text}')
            print_message(f'[{counter + 1}/{num_samples}] BLEU-1    : {batch_bleu_scores[0]:0.3f}')
            print_message(f'[{counter + 1}/{num_samples}] BLEU-2    : {batch_bleu_scores[1]:0.3f}')
            print_message(f'[{counter + 1}/{num_samples}] BLEU-3    : {batch_bleu_scores[2]:0.3f}')
            print_message(f'[{counter + 1}/{num_samples}] BLEU-4    : {batch_bleu_scores[3]:0.3f}')

            counter += 1
            if counter == num_samples:
                break

    if writer is not None:
        eval_bleu_scores = []
        for n_gram in range(1, 5):
            score = bleu_score(
                predicted_texts,
                target_texts,
                max_n=n_gram,
                weights=[1 / n_gram] * n_gram
            )
            eval_bleu_scores.append(score)

        writer.add_scalars('eval_BLEU', {
            'BLEU-1': eval_bleu_scores[0],
            'BLEU-2': eval_bleu_scores[1],
            'BLEU-3': eval_bleu_scores[2],
            'BLEU-4': eval_bleu_scores[3],
        }, global_step=epoch)
        writer.flush()

        print_message(f'Evaluation BLEU-1: {eval_bleu_scores[0]:0.3f}')
        print_message(f'Evaluation BLEU-2: {eval_bleu_scores[1]:0.3f}')
        print_message(f'Evaluation BLEU-3: {eval_bleu_scores[2]:0.3f}')
        print_message(f'Evaluation BLEU-4: {eval_bleu_scores[3]:0.3f}')

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
    learning_rate = config['learning_rate']
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    initial_epoch = 0
    global_step = 0
    _preload = config['preload']
    if _preload is not None:
        model_filename = model_util.get_weights_file_path(epoch=f'{_preload:0>2}', config=config)
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

            # backpropagate the loss
            loss.backward()

            # clipping the gradient
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=config['grad_clipping'])

            # update weights and learning rate
            optimizer.step()
            optimizer.zero_grad()

            running_loss += loss.item()
            epoch_loss += loss.item()

            if isinstance(log_step, int) and (batch_idx + 1) % log_step == 0:
                # run validation
                evaluate_model(
                    model,
                    device,
                    validation_data_loader,
                    src_tokenizer,
                    target_tokenizer,
                    config['seq_length'],
                    epoch,
                    batch_message_printer,
                    beam_size=config['beam_size'],
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
                epoch,
                batch_message_printer,
                beam_size=config['beam_size'],
                writer=writer,
                num_samples=config['num_eval_samples'],
            )
            writer.add_scalar('loss/epoch_loss', epoch_loss / len(batch_iterator), epoch)
            writer.flush()

        # save the model after every epoch
        model_filename = model_util.get_weights_file_path(f'{epoch:02d}', config)
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
