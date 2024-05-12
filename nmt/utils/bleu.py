from tqdm.autonotebook import tqdm

import torch
from torch import Tensor
# from torchtext.data.metrics import bleu_score

from tokenizers import Tokenizer

import evaluate

from transformer import Transformer
from nmt.billingual_dataset import BilingualDataset
from nmt.utils import (
    model as model_util,
    misc as misc_util,
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
    log_sentences: bool = False,
    logging_interval: int = 20,
    max_steps: int | None = None,
) -> float:

    device = model.device
    src_text_list = []
    target_text_list = []
    pred_text_list = []

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

    sacrebleu = evaluate.load('sacrebleu')

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
                    cand = misc_util.remove_end_tokens(cand, target_tokenizer, contains_id=True)

                    cand_text_list.append(target_tokenizer.decode(cand, skip_special_tokens=False))
            pred_token_ids = misc_util.remove_end_tokens(pred_token_ids, target_tokenizer, contains_id=True)

            # retrieve src_tokens and target_tokens
            encoder_input_eos_index = misc_util.tensor_find_value(
                encoder_input,
                src_tokenizer.token_to_id(SpecialToken.EOS),
                kth=0,
            )
            labels_eos_index = misc_util.tensor_find_value(
                labels,
                target_tokenizer.token_to_id(SpecialToken.EOS),
                kth=0,
            )
            assert encoder_input_eos_index is not None and encoder_input_eos_index.ndim == 1
            assert labels_eos_index is not None and labels_eos_index.ndim == 1
            assert encoder_input[0].item() == src_tokenizer.token_to_id(SpecialToken.SOS)

            # note that `src_tokens`, `target_tokens`, and `pred_tokens`
            # contains no end tokens (i.e. <SOS> and <EOS>)
            src_token_ids = encoder_input[1:encoder_input_eos_index.item()].detach().cpu().numpy()
            target_token_ids = labels[:labels_eos_index.item()].detach().cpu().numpy()
            if isinstance(pred_token_ids, Tensor):
                pred_token_ids = pred_token_ids.detach().cpu().numpy()

            src_text = src_tokenizer.decode(src_token_ids, skip_special_tokens=False)
            target_text = target_tokenizer.decode(target_token_ids, skip_special_tokens=False)
            pred_text = target_tokenizer.decode(pred_token_ids, skip_special_tokens=False)

            src_text_list.append(src_text)
            target_text_list.append([target_text])
            pred_text_list.append(pred_text)

            if log_sentences and data_idx % logging_interval == 0:
                bleu_score = sacrebleu.compute(predictions=[pred_text], references=[[target_text]])

                # TODO: using write in nested tqdm loop is hard to read
                dataset_iterator.write(f'Source: {src_text}')
                dataset_iterator.write(f'Target: {target_text}')
                if cand_text_list is not None:
                    for cand_text_idx, cand_text in enumerate(cand_text_list):
                        dataset_iterator.write(f'Predicted-{cand_text_idx + 1}: {cand_text}')
                else:
                    dataset_iterator.write(f'Predicted: {" ".join(pred_text)}')

                dataset_iterator.write(f'BLEU: {bleu_score["score"]:0.3f}')

    dataset_blue_score = sacrebleu.compute(
        predictions=pred_text_list,
        references=target_text_list,
    )

    # set model back to training mode
    model.train()

    return dataset_blue_score['score']
