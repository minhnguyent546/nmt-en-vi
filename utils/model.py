from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as Fun
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from tokenizers import Tokenizer

from transformer import Transformer, make_transformer
from transformer.utils.functional import create_causal_mask
import constants as const

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
        attention_dropout_rate=config['attention_dropout_rate'],
    )
    return model

def get_weights_file_path(epoch: str, config: dict) -> str:
    model_dir = Path(config['checkpoints_dir']) / config['model_dir']
    model_basename = config['model_basename']
    model_file = f'{model_basename}_{epoch}.pt'
    return str(model_dir / model_file)

def get_latest_weights_file_path(config: dict) -> str | None:
    model_dir = Path(config['checkpoints_dir']) / config['model_dir']
    model_basename = config['model_basename']
    saved_files = list(model_dir.glob(f'{model_basename}_*.pt'))
    if len(saved_files) > 0:
        latest_file = sorted(saved_files)[-1]
        return str(latest_file)

    print(f'No model weights found at {model_dir}')
    return None

def noam_decay_lr(step_num: int, d_model: int = 512, warmup_steps: int = 4000):
    """
    As described in https://arxiv.org/pdf/1706.03762.pdf
    """
    step_num = max(step_num, 1)
    return d_model ** (-0.5) * min(step_num ** (-0.5), step_num * warmup_steps ** (-1.5))

def decode_with_teacher_forcing(
    model: Transformer,
    device: torch.device,
    encoder_input: Tensor,
    decoder_input: Tensor,
    encoder_mask: Tensor,
    decoder_mask: Tensor,
    has_batch_dim: bool = False,
) -> Tensor:
    """
    Args:
        model (Transformer): model to be used for decoding
        device (torch.device): device
        encoder_input (Tensor): encoder input
        decoder_input (Tensor): decoder input
        encoder_mask (Tensor): mask tensor for encoder input
        decoder_mask (Tensor): mask tensor for decoder input
        has_batch_dim (bool): whether input tensors have batch dimension (default: False)

    Returns:
        Tensor: tensor of predicted token ids
    """
    encoder_input = encoder_input.to(device)
    decoder_input = decoder_input.to(device)
    encoder_mask = encoder_mask.to(device)
    decoder_mask = decoder_mask.to(device)

    if not has_batch_dim:
        encoder_input.unsqueeze_(0)
        decoder_input.unsqueeze_(0)

    encoder_output = model.encode(encoder_input, encoder_mask)  # (batch_size, seq_length, d_model)
    decoder_output = model.decode(encoder_output, decoder_input, encoder_mask, decoder_mask)  # (batch_size, seq_length, d_model)
    logits = model.linear(decoder_output)  # (batch_size, seq_length, target_vocab_size)

    pred_token_ids = logits.argmax(dim=-1)  # (batch_size, seq_length)

    return pred_token_ids

def greedy_search_decode(
    model: Transformer,
    device: torch.device,
    encoder_input: Tensor,
    encoder_mask: Tensor,
    target_tokenizer: Tokenizer,
    seq_length: int,
) -> Tensor:
    """
    Args:
        model (Transformer): model to be used for decoding
        device (torch.device): device
        encoder_input (Tensor): encoder input
        encoder_mask (Tensor): mask tensor for encoder input
        target_tokenizer (Tokenizer): target tokenizer
        seq_lenght (int): maximum sequence length

    Returns:
        Tensor: tensor of predicted token ids
    """

    sos_token_id = target_tokenizer.token_to_id(const.SOS_TOKEN)
    eos_token_id = target_tokenizer.token_to_id(const.EOS_TOKEN)

    encoder_input = encoder_input.unsqueeze(0).to(device)
    encoder_mask = encoder_mask.unsqueeze(0).to(device)
    encoder_output = model.encode(encoder_input, encoder_mask)

    # initialize decoder input which contains only <SOS> token
    decoder_input = torch.empty((1, 1)).fill_(sos_token_id).type_as(encoder_input).to(device)
    for _ in range(seq_length):
        # create mask for decoder input
        decoder_mask = create_causal_mask(decoder_input.size(1)).type_as(encoder_mask).to(device)

        # decode
        decoder_output = model.decode(encoder_output, decoder_input, encoder_mask, decoder_mask)

        # get token with highest probability
        logits = model.linear(decoder_output[:, -1, :])  # (1, target_vocab_size)
        next_token = logits.argmax(dim=-1)

        # concatenate the next token to the decoder input for the next prediction
        decoder_input = torch.cat([
            decoder_input,
            torch.empty((1, 1)).fill_(next_token.item()).type_as(encoder_input).to(device)
        ], dim=1)

        # if we reach the <EOS> token, then stop
        if next_token == eos_token_id:
            break

    return decoder_input.squeeze(0)

def length_penalty(length: int, alpha: float = 0.6) -> float:
    """
    As formula described in We at al. (2016)
    """
    return (5 + length) ** alpha / (5 + 1) ** alpha

def beam_search_decode(
    model: Transformer,
    device: torch.device,
    beam_size: int,
    encoder_input: Tensor,
    encoder_mask: Tensor,
    target_tokenizer: Tokenizer,
    seq_length: int,
) -> Tensor:
    """
    Args:
        model (Transformer): model to be used for decoding
        device (torch.device): device
        beam_size (int): beam size
        encoder_input (Tensor): encoder input
        encoder_mask (Tensor): mask tensor for encoder input
        target_tokenizer (Tokenizer): target tokenizer
        seq_lenght (int): maximum sequence length

    Returns:
        Tensor: tensor of predicted token ids
    """

    sos_token_id = target_tokenizer.token_to_id(const.SOS_TOKEN)
    eos_token_id = target_tokenizer.token_to_id(const.EOS_TOKEN)

    encoder_input = encoder_input.unsqueeze(0).to(device)
    encoder_mask = encoder_mask.unsqueeze(0).to(device)
    encoder_output = model.encode(encoder_input, encoder_mask)

    # initialize decoder input which contains only <SOS> token
    decoder_input = torch.empty((1, 1)).fill_(sos_token_id).type_as(encoder_input).to(device)

    # candidate list of tuples (decoder_input, log_score)
    cands = [(decoder_input, 0.0)]
    for _ in range(seq_length):
        new_cands = []

        for cand, log_score in cands:
            # do not expand the candidate that have reached <EOS> token
            if cand[0, -1].item() == eos_token_id:
                new_cands.append((cand, log_score))
                continue

            # create mask for decoder input
            cand_mask = create_causal_mask(cand.size(1)).type_as(encoder_mask).to(device)

            # decode
            decoder_output = model.decode(encoder_output, cand, encoder_mask, cand_mask)

            # get next token probabilities
            # logits: shape ``(1, target_vocab_size)``
            # topk_prob       : shape ``(1, beam_size)``
            # topk_token      : shape ``(1, beam_size)``
            logits = model.linear(decoder_output[:, -1, :])

            output = Fun.log_softmax(logits, dim=-1) / length_penalty(cand.size(1) + 1)
            # get the top k largest tokens
            topk_token_prob, topk_token = torch.topk(output, beam_size, dim=1)
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

def train(
    model: Transformer,
    device: torch.device,
    optimizer,
    loss_function,
    train_data_loader: DataLoader,
    epoch: int,
    global_step: int,
    config: dict,
    train_max_steps: int | None = None,
    writer: SummaryWriter | None = None,
    lr_scheduler = None,
) -> dict:
    """
    Args:
        model (Transformer): model to be trained
        device (device): device
        optimizer: optimizer
        loss_function: loss function
        train_data_loader (DataLoader): data loader for training
        epoch (int): current epoch
        global_step (int): start from this global step
        config (dict): dictionary of configurations
        train_max_steps (int): maximum number of iterations for training (default: None)
        writer (SummaryWriter): tensorboard writer (default: None)
        lr_scheduler: learning rate scheduler (default: None)

    Returns:
        train stats (dict)
    """

    num_epochs = config['num_epochs']
    total_steps = len(train_data_loader)
    if train_max_steps is not None:
        total_steps = min(total_steps, train_max_steps)

    batch_iterator = tqdm(train_data_loader,
                          desc=f'Processing epoch {epoch + 1:02d}/{num_epochs:02d}',
                          total=total_steps)
    train_loss = 0.0

    # set model in training mode
    model.train()

    for batch_idx, batch in enumerate(batch_iterator):
        if batch_idx >= total_steps:
            break

        encoder_input = batch['encoder_input'].to(device)  # (batch_size, seq_length)
        decoder_input = batch['decoder_input'].to(device)  # (batch_size, seq_length)
        encoder_mask = batch['encoder_mask'].to(device)  # (batch_size, 1, 1, seq_length)
        decoder_mask = batch['decoder_mask'].to(device)  # (batch_size, 1, seq_length, seq_length)

        encoder_output = model.encode(encoder_input, encoder_mask)  # (batch_size, seq_length, d_model)
        decoder_output = model.decode(encoder_output, decoder_input, encoder_mask, decoder_mask)  # (batch_size, seq_length, d_model)
        logits = model.linear(decoder_output)  # (batch_size, seq_length, target_vocab_size)
        labels = batch['labels'].to(device)  # (batch_size, seq_length)

        # calculate the loss
        # logits: (batch_size * seq_length, target_vocab_size)
        # label: (batch_size * seq_length)
        target_vocab_size = logits.size(-1)
        loss = loss_function(logits.view(-1, target_vocab_size), labels.view(-1))

        # backpropagate the loss
        loss.backward()

        if config['max_grad_norm'] > 0:
            # clipping the gradient
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=config['max_grad_norm'])

        # update weights and learning rate
        optimizer.step()
        optimizer.zero_grad()
        if lr_scheduler is not None:
            if writer is not None:
                for group_id, group_lr in enumerate(lr_scheduler.get_last_lr()):
                    writer.add_scalar(f'learning_rate/group-{group_id}', group_lr, global_step)
            lr_scheduler.step()

        train_loss += loss.item()
        batch_iterator.set_postfix({'loss': f'{loss.item():0.3f}'})

        if writer is not None:
            writer.add_scalar('loss/train_batch_loss', loss.item(), global_step)
            writer.flush()

        global_step += 1

    return {
        'train_loss': train_loss / total_steps,
        'global_step': global_step,
    }

def validate(
    model: Transformer,
    device: torch.device,
    loss_function,
    val_data_loader: DataLoader,
    val_max_steps: int | None = None,
) -> dict:
    """
    Args:
        model (Transformer): model to be validated
        device (device): device
        loss_function: loss function
        val_data_loader (DataLoader): data loader for validation
        val_max_step (int): maximum number of iterations for validation (default: None)

    Returns:
        validation stats (dict)
    """

    val_loss = 0.0
    total_steps = len(val_data_loader)
    if val_max_steps is not None:
        total_steps = min(total_steps, val_max_steps)

    batch_iterator = tqdm(val_data_loader,
                          desc='Evaluating',
                          total=total_steps)

    # set model in validation mode
    model.eval()

    with torch.no_grad():
        for batch_idx, batch in enumerate(batch_iterator):
            if batch_idx >= total_steps:
                break

            encoder_input = batch['encoder_input'].to(device)  # (batch_size, seq_length)
            decoder_input = batch['decoder_input'].to(device)  # (batch_size, seq_length)
            encoder_mask = batch['encoder_mask'].to(device)  # (batch_size, 1, 1, seq_length)
            decoder_mask = batch['decoder_mask'].to(device)  # (batch_size, 1, seq_length, seq_length)
            labels = batch['labels'].to(device)  # (batch_size, seq_length)

            encoder_output = model.encode(encoder_input, encoder_mask)  # (batch_size, seq_length, d_model)
            decoder_output = model.decode(encoder_output, decoder_input, encoder_mask, decoder_mask)  # (batch_size, seq_length, d_model)
            logits = model.linear(decoder_output)  # (batch_size, seq_length, target_vocab_size)

            # calculating the loss
            # logits: (batch_size * seq_length, target_vocab_size)
            # label: (batch_size * seq_length)
            target_vocab_size = logits.size(-1)
            loss = loss_function(logits.view(-1, target_vocab_size), labels.view(-1))
            val_loss += loss.item()

            batch_iterator.set_postfix({'loss': f'{loss.item():0.3f}'})

    return {
        'val_loss': val_loss / total_steps,
    }
