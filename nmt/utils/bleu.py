from tqdm.autonotebook import tqdm

import numpy as np

import torch
from torch import Tensor
from torchtext.data.metrics import bleu_score

from tokenizers import Tokenizer

from transformer import Transformer
from nmt.billingual_dataset import BilingualDataset
from nmt.utils import (
    model as model_util,
    misc as misc_uitl,
)
from nmt.constants import SpecialToken

def compute_dataset_bleu(
    model: Transformer,
    dataset: BilingualDataset,
    src_tokenizer: Tokenizer,
    target_tokenizer: Tokenizer,
    seq_length: int,
    teacher_forcing: bool = False,
    beam_size: int | None = None,
    beam_return_topk: int = 1,
    max_n: int = 4,
    log_sentences: bool = False,
    logging_interval: int = 20,
    max_steps: int | None = None,
) -> list[float]:

    device = model.device
    src_tokens_list = []
    target_tokens_list = []
    pred_tokens_list = []

    total_steps = len(dataset)
    if max_steps is not None:
        total_steps = min(total_steps, max_steps)

    dataset_iterator = tqdm(enumerate(dataset),
                            desc='Computing validation BLEU',
                            total=total_steps)

    cand_list = None
    cand_text_list = None

    # set model in evaluation mode
    model.eval()

    with torch.no_grad():
        for data_idx, data in dataset_iterator:
            if data_idx >= total_steps:
                break

            encoder_input = data['encoder_input']
            labels = data['labels']

            if teacher_forcing:
                # decoding with teacher forcing
                decoder_input = data['decoder_input']

                pred_token_ids = model_util.decode_with_teacher_forcing(model, device, encoder_input,
                                                                        decoder_input, has_batch_dim=False)
                pred_token_ids = pred_token_ids[0]
            elif beam_size is not None and beam_size > 1:
                # decoding with beam search
                cand_list = model_util.beam_search_decode(model, device, beam_size, encoder_input,
                                                          target_tokenizer, seq_length, return_topk=beam_return_topk)
                pred_token_ids = cand_list[0]
            else:
                # decoding with greedy search
                pred_token_ids = model_util.greedy_search_decode(model, device, encoder_input,
                                                                 target_tokenizer, seq_length)

            if cand_list is not None:
                cand_text_list = []
                for cand in cand_list:
                    cand = cand.detach().cpu().numpy()

                    # remove <SOS> and <EOS> tokens if they are present
                    cand = remove_end_tokens(cand, target_tokenizer, contains_id=True)

                    cand_text_list.append(target_tokenizer.decode(cand, skip_special_tokens=False))
            pred_token_ids = remove_end_tokens(pred_token_ids, target_tokenizer, contains_id=True)

            # retrieve src_tokens and target_tokens
            encoder_input_eos_index = misc_uitl.tensor_find_value(
                encoder_input,
                src_tokenizer.token_to_id(SpecialToken.EOS),
                kth=0,
            )
            labels_eos_index = misc_uitl.tensor_find_value(
                labels,
                target_tokenizer.token_to_id(SpecialToken.EOS),
                kth=0,
            )
            assert encoder_input_eos_index is not None and encoder_input_eos_index.ndim == 1
            assert labels_eos_index is not None and labels_eos_index.ndim == 1
            assert encoder_input[0].item() == src_tokenizer.token_to_id(SpecialToken.SOS)

            # note that `src_tokens`, `target_tokens`, and `pred_tokens`
            # contains no end tokens (i.e. <SOS> and <EOS>)
            src_tokens = [src_tokenizer.id_to_token(token_id) for token_id in encoder_input[1:encoder_input_eos_index.item()]]
            target_tokens = [target_tokenizer.id_to_token(token_id) for token_id in labels[:labels_eos_index.item()]]
            pred_tokens = [target_tokenizer.id_to_token(token_id) for token_id in pred_token_ids]

            src_tokens_list.append(src_tokens)
            target_tokens_list.append([target_tokens])
            pred_tokens_list.append(pred_tokens)

            if log_sentences and data_idx % logging_interval == 0:
                bleu_scores = [
                    bleu_score(candidate_corpus=[pred_tokens],
                               references_corpus=[[target_tokens]],
                               max_n=n_gram,
                               weights=[1 / n_gram] * n_gram)
                    for n_gram in range(1, max_n + 1)
                ]

                # TODO: using write in nested tqdm loop is hard to read
                dataset_iterator.write(f'Source: {" ".join(src_tokens)}')
                dataset_iterator.write(f'Target: {" ".join(target_tokens)}')
                if cand_text_list is not None:
                    for cand_text_idx, cand_text in enumerate(cand_text_list):
                        dataset_iterator.write(f'Predicted-{cand_text_idx + 1}: {cand_text}')
                else:
                    dataset_iterator.write(f'Predicted: {" ".join(pred_tokens)}')
                for n_gram in range(1, max_n + 1):
                    dataset_iterator.write(f'BLEU-{n_gram}: {bleu_scores[n_gram - 1]:0.3f}')

    bleu_scores = [
        bleu_score(candidate_corpus=pred_tokens_list,
                   references_corpus=target_tokens_list,
                   max_n=n_gram,
                   weights=[1 / n_gram] * n_gram)
        for n_gram in range(1, max_n + 1)
    ]

    # set model back to training mode
    model.train()

    return bleu_scores

def remove_end_tokens(
    tokens: Tensor | np.ndarray,
    tokenizer: Tokenizer,
    *,
    contains_id: bool = False
) -> Tensor | np.ndarray:
    assert tokens.ndim == 1
    if isinstance(tokens, Tensor) and tokens.numel() == 0:
        return tokens
    if isinstance(tokens, np.ndarray) and tokens.size == 0:
        return tokens

    first_token = tokenizer.id_to_token(tokens[0]) if contains_id else tokens[0]
    last_token = tokenizer.id_to_token(tokens[-1]) if contains_id else tokens[-1]
    if first_token == SpecialToken.SOS:
        tokens = tokens[1:]
    if last_token == SpecialToken.EOS:
        tokens = tokens[:-1]
    return tokens
