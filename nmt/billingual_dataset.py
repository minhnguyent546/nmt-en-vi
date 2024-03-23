import torch
from torch import Tensor
from torch.utils.data import Dataset

from tokenizers import Tokenizer

import nmt.constants as const

class BilingualDataset(Dataset):
    def __init__(
        self,
        dataset,
        src_tokenizer: Tokenizer,
        target_tokenizer: Tokenizer,
        src_lang: str,
        target_lang: str,
        seq_length: int,
        add_padding_tokens: bool = False,
    ):
        super().__init__()
        self.dataset = dataset
        self.src_tokenizer = src_tokenizer
        self.target_tokenizer = target_tokenizer
        self.src_lang = src_lang
        self.target_lang = target_lang
        self.seq_length = seq_length
        self.add_padding_tokens = add_padding_tokens

        assert src_tokenizer.token_to_id(const.SOS_TOKEN) == target_tokenizer.token_to_id(const.SOS_TOKEN)
        assert src_tokenizer.token_to_id(const.EOS_TOKEN) == target_tokenizer.token_to_id(const.EOS_TOKEN)
        assert src_tokenizer.token_to_id(const.PAD_TOKEN) == target_tokenizer.token_to_id(const.PAD_TOKEN)

        self.sos_token_id = src_tokenizer.token_to_id(const.SOS_TOKEN)
        self.eos_token_id = src_tokenizer.token_to_id(const.EOS_TOKEN)
        self.pad_token_id = src_tokenizer.token_to_id(const.PAD_TOKEN)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        src_text = self.dataset[index]['translation'][self.src_lang]
        target_text = self.dataset[index]['translation'][self.target_lang]

        encode_input_tokens = self.src_tokenizer.encode(src_text).ids
        decode_input_tokens = self.target_tokenizer.encode(target_text).ids

        encode_num_paddings, decode_num_paddings = 0, 0
        if self.add_padding_tokens:
            encode_num_paddings = self.seq_length - len(encode_input_tokens) - 2  # exclude <SOS> & <EOS>
            decode_num_paddings = self.seq_length - len(decode_input_tokens) - 1  # exclude <SOS> | <EOS>

        assert encode_num_paddings >= 0, "The length of the source text is too long"
        assert decode_num_paddings >= 0, "The length of the target text is too long"

        encoder_input = torch.cat([
            Tensor([self.sos_token_id]),
            Tensor(encode_input_tokens),
            Tensor([self.eos_token_id]),
            Tensor([self.pad_token_id]).repeat(encode_num_paddings)
        ]).type(torch.int32)
        decoder_input = torch.cat([
            Tensor([self.sos_token_id]),
            Tensor(decode_input_tokens),
            Tensor([self.pad_token_id]).repeat(decode_num_paddings)
        ]).type(torch.int32)
        labels = torch.cat([
            Tensor(decode_input_tokens),
            Tensor([self.eos_token_id]),
            Tensor([self.pad_token_id]).repeat(decode_num_paddings)
        ]).type(torch.int64)  # int32 has a problem with nll loss forward on cuda

        if self.add_padding_tokens:
            assert encoder_input.size(0) == self.seq_length
            assert decoder_input.size(0) == self.seq_length
            assert labels.size(0) == self.seq_length

        return {
            'src_text': src_text,
            'target_text': target_text,
            'encoder_input': encoder_input,
            'decoder_input': decoder_input,
            'labels': labels,
        }
