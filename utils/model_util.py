import sys
sys.path.append('../')

import torch
import torch.nn.functional as Fun
from torchtext.data.metrics import bleu_score
from torch import Tensor, device
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from tokenizers import Tokenizer

from pathlib import Path

from transformer import Transformer, make_transformer
import utils.model_util as model_util
import constants as const

def count_parameters(model) -> int:
    return sum(
        param.numel() for param in model.parameters() if param.requires_grad
    )

def create_mask(seq_length: int) -> Tensor:
    tril_mask = torch.tril(torch.ones(1, seq_length, seq_length)).bool() # (1, seq_length, seq_length)
    return tril_mask

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

def cal_bleu_score(
    candidate_corpus,
    references_corpus,
    max_n: int | list[int] = 4,
    weights=[0.25] * 4
) -> float | list[float]:
    if isinstance(max_n, int):
        return bleu_score(candidate_corpus, references_corpus, max_n=max_n, weights=weights)

    scores = []
    for n_gram in max_n:
        cur_score = bleu_score(candidate_corpus, references_corpus, max_n=n_gram, weights=[1 / n_gram] * n_gram)
        scores.append(cur_score)

    return scores

def greedy_search_decode(
    model: Transformer,
    device: device,
    src: Tensor,
    src_mask: Tensor,
    target_tokenizer: Tokenizer,
    seq_length: int
) -> Tensor:
    # assume batch_size is 1
    sos_token_id = target_tokenizer.token_to_id(const.SOS_TOKEN)
    eos_token_id = target_tokenizer.token_to_id(const.EOS_TOKEN)

    encoder_output = model.encode(src, src_mask) # (batch_size, seq_length, d_model)

    # initialize decoder input that contains only <SOS> token
    decoder_input = torch.empty((1, 1)).fill_(sos_token_id).type_as(src).to(device)
    for _ in range(seq_length):
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
    As formula described in We at al. (2016)
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
    sos_token_id = target_tokenizer.token_to_id(const.SOS_TOKEN)
    eos_token_id = target_tokenizer.token_to_id(const.EOS_TOKEN)

    encoder_output = model.encode(src, src_mask) # (batch_size, seq_length, d_model)

    # initialize decoder input that contains only <SOS> token
    decoder_input = torch.empty((1, 1)).fill_(sos_token_id).type_as(src).to(device)

    # candidate list of tuple (decoder_input, log_score)
    cands = [(decoder_input, 0.0)]
    for _ in range(seq_length):
        new_cands = []

        for cand, log_score in cands:
            # do not expand the candidate that have reached <EOS> token
            if cand[0, -1].item() == eos_token_id:
                new_cands.append((cand, log_score))
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
    data_loader: DataLoader,
    src_tokenizer: Tokenizer,
    target_tokenizer: Tokenizer,
    seq_length: int,
    print_message,
    beam_size: int | None = None,
    epoch: int | None = None,
    writer: SummaryWriter | None = None,
    num_samples: int = -1,
) -> None:
    model.eval()
    counter = 0

    src_texts = []
    target_texts = []
    predicted_texts = []
    if num_samples == -1:
        num_samples = len(data_loader)

    with torch.no_grad():
        # environment with no gradient calculation
        ignored_tokens = [const.UNK_TOKEN]
        for batch in data_loader:
            encoder_input = batch['encoder_input'].to(device) # (batch_size, seq_length)
            encoder_mask = batch['encoder_mask'].to(device) # (batch_size, 1, 1, seq_length)

            batch_size = encoder_input.size(0)
            assert batch_size == 1, 'batch_size must be 1 for evaluation'

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

            batch_bleu_scores = cal_bleu_score([predicted_text_tokens], [[target_text_tokens]], max_n=[1, 2, 3, 4])

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

    eval_bleu_scores = cal_bleu_score(predicted_texts, target_texts, max_n=[1, 2, 3, 4])

    print_message(f'Evaluation BLEU-1: {eval_bleu_scores[0]:0.3f}')
    print_message(f'Evaluation BLEU-2: {eval_bleu_scores[1]:0.3f}')
    print_message(f'Evaluation BLEU-3: {eval_bleu_scores[2]:0.3f}')
    print_message(f'Evaluation BLEU-4: {eval_bleu_scores[3]:0.3f}')

    if writer is not None:
        assert epoch is not None, 'epoch must be provided when writer is not None'
        writer.add_scalars('eval_BLEU', {
            'BLEU-1': eval_bleu_scores[0],
            'BLEU-2': eval_bleu_scores[1],
            'BLEU-3': eval_bleu_scores[2],
            'BLEU-4': eval_bleu_scores[3],
        }, global_step=epoch)
        writer.flush()
