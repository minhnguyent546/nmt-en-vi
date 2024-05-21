from tqdm.autonotebook import tqdm

import torch
from torch import Tensor

import evaluate
from tokenizers import Tokenizer

from nmt.billingual_dataset import BilingualDataset
from nmt.constants import LOWER_ONE_EIGHTH_BLOCK, SpecialToken
from nmt.utils import misc as misc_util, model as model_util
from transformer import Transformer


@torch.no_grad()
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
    **kwargs,
) -> float:

    device = model.device
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
    is_training = model.training
    model.eval()

    sacrebleu = evaluate.load('sacrebleu')

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
            cand_text_list = []
            for cand in cand_list:
                cand = cand.detach().cpu().numpy()

                # remove <SOS> and <EOS> tokens if they are present
                cand = misc_util.remove_end_tokens(cand, target_tokenizer, contains_id=True)

                cand_text = target_tokenizer.decode(cand).replace('_', LOWER_ONE_EIGHTH_BLOCK)
                cand_text_list.append(cand_text)

            pred_token_ids = cand_list[0]
        else:
            # decoding with greedy search
            pred_token_ids = model_util.greedy_search_decode(model, device, encoder_input,
                                                             target_tokenizer, seq_length)

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

        # tokenizer.decode method will remove special tokens by default (e.g. <UNK>)
        # it should be, because keep <UNK> tokens will increase the BLEU score
        # but has no meaning. See Post, 2018
        src_text = src_tokenizer.decode(src_token_ids)
        target_text = target_tokenizer.decode(target_token_ids).replace('_', LOWER_ONE_EIGHTH_BLOCK)
        pred_text = target_tokenizer.decode(pred_token_ids).replace('_', LOWER_ONE_EIGHTH_BLOCK)

        target_text_list.append(target_text)
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
    model.train(is_training)

    return dataset_blue_score['score']
