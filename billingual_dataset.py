import torch
from torch import Tensor
from torch.utils.data import Dataset

from tokenizers import Tokenizer

from transformer.utils.functional import create_causal_mask
import constants as const

class BilingualDataset(Dataset):
    def __init__(
        self,
        dataset,
        src_tokenizer: Tokenizer,
        target_tokenizer: Tokenizer,
        src_lang: str,
        target_lang: str,
        seq_length: int
    ):
        super().__init__()
        self.dataset = dataset
        self.src_tokenizer = src_tokenizer
        self.target_tokenizer = target_tokenizer
        self.src_lang = src_lang
        self.target_lang = target_lang
        self.seq_length = seq_length

        assert src_tokenizer.token_to_id(const.SOS_TOKEN) == target_tokenizer.token_to_id(const.SOS_TOKEN)
        assert src_tokenizer.token_to_id(const.EOS_TOKEN) == target_tokenizer.token_to_id(const.EOS_TOKEN)
        assert src_tokenizer.token_to_id(const.PAD_TOKEN) == target_tokenizer.token_to_id(const.PAD_TOKEN)

        sos_token_id = src_tokenizer.token_to_id(const.SOS_TOKEN)
        eos_token_id = src_tokenizer.token_to_id(const.EOS_TOKEN)
        pad_token_id = src_tokenizer.token_to_id(const.PAD_TOKEN)

        self.sos_token = Tensor([sos_token_id]).type(torch.int64)
        self.eos_token = Tensor([eos_token_id]).type(torch.int64)
        self.pad_token = Tensor([pad_token_id]).type(torch.int64)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        src_text = self.dataset[index]['translation'][self.src_lang]
        target_text = self.dataset[index]['translation'][self.target_lang]

        encode_input_tokens = self.src_tokenizer.encode(src_text).ids
        decode_input_tokens = self.target_tokenizer.encode(target_text).ids

        encode_num_paddings = self.seq_length - len(encode_input_tokens) - 2  # exclude <SOS> & <EOS>
        decode_num_paddings = self.seq_length - len(decode_input_tokens) - 1  # exclude <SOS> | <EOS>

        assert encode_num_paddings >= 0, "The length of the source text is too long"
        assert decode_num_paddings >= 0, "The length of the target text is too long"

        encoder_input = torch.cat([
            self.sos_token,
            Tensor(encode_input_tokens).type(torch.int64),
            self.eos_token,
            self.pad_token.repeat(encode_num_paddings)
        ])
        decoder_input = torch.cat([
            self.sos_token,
            Tensor(decode_input_tokens).type(torch.int64),
            self.pad_token.repeat(decode_num_paddings)
        ])
        labels = torch.cat([
            Tensor(decode_input_tokens).type(torch.int64),
            self.eos_token,
            self.pad_token.repeat(decode_num_paddings)
        ])
        assert encoder_input.size(0) == self.seq_length
        assert decoder_input.size(0) == self.seq_length
        assert labels.size(0) == self.seq_length

        encoder_mask = (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0)  # (1, 1, seq_length)
        causal_mask = create_causal_mask(self.seq_length).unsqueeze(0)
        decoder_mask = (decoder_input != self.pad_token) & causal_mask  # (1, seq_length, seq_length)

        return {
            'src_text': src_text,
            'target_text': target_text,
            'encoder_input': encoder_input,
            'decoder_input': decoder_input,
            'encoder_mask': encoder_mask,
            'decoder_mask': decoder_mask,
            'labels': labels,
        }
