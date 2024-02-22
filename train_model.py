import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from tokenizers import Tokenizer

from tqdm import tqdm # progress bar helper

from pathlib import Path

from config import get_config
import utils.model_util as model_util
import constants as const

def train_model(config):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device {device}')
    device = torch.device(device)

    checkpoints_dir = Path(config['checkpoints_dir'])
    model_dir = checkpoints_dir / config['model_dir']
    model_dir.mkdir(parents=True, exist_ok=True)

    print('Loading data loaders')
    data_loaders = torch.load(checkpoints_dir / config['data_loaders_basename'])
    train_data_loader = data_loaders['train']
    validation_data_loader = data_loaders['validation']

    print('Loading tokenizers')
    src_tokenizer = Tokenizer.from_file(str(checkpoints_dir / config['tokenizer_basename'].format(config['src_lang'])))
    target_tokenizer = Tokenizer.from_file(str(checkpoints_dir / config['tokenizer_basename'].format(config['target_lang'])))

    src_vocab_size, target_vocab_size = src_tokenizer.get_vocab_size(), target_tokenizer.get_vocab_size()
    model = model_util.make_model(src_vocab_size, target_vocab_size, config)
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
        print(f'Loading weights from epoch {_preload:0>2}')
        states = torch.load(model_filename)

        # continue from previous completed epoch
        initial_epoch = states['epoch'] + 1
        
        model.load_state_dict(states['model_state_dict'])
        optimizer.load_state_dict(states['optimizer_state_dict'])
        global_step = states['global_step']

    loss_function = nn.CrossEntropyLoss(ignore_index=src_tokenizer.token_to_id(const.PAD_TOKEN),
                                        label_smoothing=config['label_smoothing'])

    num_epochs = config['num_epochs']
    for epoch in range(initial_epoch, num_epochs):
        # clear cuda cache
        torch.cuda.empty_cache()

        model.train()

        epoch_loss = 0.0
        batch_iterator = tqdm(train_data_loader, desc=f'Processing epoch {epoch + 1:02d}/{num_epochs:02d}')
        batch_message_printer = lambda message: batch_iterator.write(message)
        for batch in batch_iterator:
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
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=config['max_grad_norm'])

            # update weights and learning rate
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            global_step += 1

        model_util.evaluate_model(
            model,
            device,
            validation_data_loader,
            src_tokenizer,
            target_tokenizer,
            config['seq_length'],
            batch_message_printer,
            beam_size=config['beam_size'],
            epoch=epoch,
            writer=writer,
            num_samples=config['num_validation_samples'],
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
    config = get_config()
    train_model(config)
