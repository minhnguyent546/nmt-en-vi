from tqdm import tqdm

import torch
# from torchtext.data.metrics import bleu_score

from tokenizers import Tokenizer

from transformer import Transformer
from nmt.billingual_dataset import BilingualDataset
from nmt.utils import model as model_util

def compute_dataset_bleu(
    model: Transformer,
    device: torch.device,
    billingual_dataset: BilingualDataset,
    target_tokenizer: Tokenizer,
    seq_length: int,
    teacher_forcing: bool = True,
    beam_size: int | None = None,
    beam_return_topk: int = 1,
    max_n: int = 4,
    log_sentences: bool = False,
    logging_interval: int = 20,
    max_steps: int | None = None,
) -> list[float]:

    src_tokens_list = []
    target_tokens_list = []
    pred_tokens_list = []

    total_steps = len(billingual_dataset)
    if max_steps is not None:
        total_steps = min(total_steps, max_steps)

    dataset_iterator = tqdm(billingual_dataset,
                            desc='Computing validation BLEU',
                            total=total_steps)
    cand_list = None
    cand_text_list = None

    # set model in evaluation mode
    model.eval()

    with torch.no_grad():
        for data_idx, data in enumerate(dataset_iterator):
            if data_idx >= total_steps:
                break

            encoder_input = data['encoder_input']
            src_text = data['src_text']
            target_text = data['target_text']

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

            pred_text = target_tokenizer.decode(pred_token_ids.detach().cpu().numpy())
            if cand_list is not None:
                cand_text_list = [
                    target_tokenizer.decode(cand.detach().cpu().numpy())
                    for cand in cand_list
                ]

            pred_tokens = target_tokenizer.encode(pred_text).tokens
            src_tokens = target_tokenizer.encode(src_text).tokens
            target_tokens = target_tokenizer.encode(target_text).tokens

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

                dataset_iterator.write(f'Source: {src_text}')
                dataset_iterator.write(f'Target: {target_text}')
                if cand_text_list is not None:
                    for cand_text_idx, cand_text in enumerate(cand_text_list):
                        dataset_iterator.write(f'Predicted-{cand_text_idx + 1}: {cand_text}')
                else:
                    dataset_iterator.write(f'Predicted: {pred_text}')
                for n_gram in range(1, max_n + 1):
                    dataset_iterator.write(f'BLEU-{n_gram}: {bleu_scores[n_gram - 1]:0.3f}')

    bleu_scores = [
        bleu_score(candidate_corpus=pred_tokens_list,
                   references_corpus=target_tokens_list,
                   max_n=n_gram,
                   weights=[1 / n_gram] * n_gram)
        for n_gram in range(1, max_n + 1)
    ]
    return bleu_scores
