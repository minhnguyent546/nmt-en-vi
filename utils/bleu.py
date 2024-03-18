from tqdm import tqdm

import torch
from torchtext.data.metrics import bleu_score

from tokenizers import Tokenizer

from transformer import Transformer
from billingual_dataset import BilingualDataset
from . import model as model_util

def compute_dataset_bleu(
    model: Transformer,
    device: torch.device,
    billingual_dataset: BilingualDataset,
    target_tokenizer: Tokenizer,
    seq_length: int,
    teacher_forcing: bool = True,
    beam_size: int | None = None,
    max_n: int = 4,
    log_sentences: bool = False,
    logging_interval: int = 20,
    max_steps: int = 200,
) -> list[float]:

    src_tokens_list = []
    target_tokens_list = []
    pred_tokens_list = []

    total_steps = min(max_steps, len(billingual_dataset))
    dataset_iterator = tqdm(billingual_dataset,
                            desc='Computing validation BLEU',
                            total=total_steps)
    # set model in evaluation mode
    model.eval()

    with torch.no_grad():
        for data_idx, data in enumerate(dataset_iterator):
            if data_idx >= total_steps:
                break

            encoder_input = data['encoder_input']
            encoder_mask = data['encoder_mask']
            src_text = data['src_text']
            target_text = data['target_text']

            if teacher_forcing:
                # decoding with teacher forcing
                decoder_input = data['decoder_input']
                decoder_mask = data['decoder_mask']

                pred_token_ids = model_util.decode_with_teacher_forcing(model, device, encoder_input, decoder_input,
                                                                        encoder_mask, decoder_mask, has_batch_dim=False)
                pred_token_ids = pred_token_ids[0]
            elif beam_size is not None and beam_size > 1:
                # decoding with beam search
                pred_token_ids = model_util.beam_search_decode(model, device, beam_size, encoder_input,
                                                               encoder_mask, target_tokenizer, seq_length)
            else:
                # decoding with greedy search
                pred_token_ids = model_util.greedy_search_decode(model, device, encoder_input,
                                                                 encoder_mask, target_tokenizer, seq_length)

            pred_text = target_tokenizer.decode(pred_token_ids.detach().cpu().numpy())
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
