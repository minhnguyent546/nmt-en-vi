import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import Dataset

from tokenizers import Tokenizer

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

        sos_token_id = src_tokenizer.token_to_id('<SOS>')
        eos_token_id = src_tokenizer.token_to_id('<EOS>')
        pad_token_id = src_tokenizer.token_to_id('<PAD>')

        self.sos_token = Tensor([sos_token_id]).type(torch.int64)
        self.eos_token = Tensor([sos_token_id]).type(torch.int64)
        self.pad_token = Tensor([sos_token_id]).type(torch.int64)

    def __len__(self):
        return len(self.dataset)

    def create_masks(self, encoder_input: Tensor, decoder_input: Tensor) -> tuple[Tensor, Tensor]:
        encoder_mask = (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() # (1, 1, seq_length)
        tril_mask = torch.tril(torch.ones(1, self.seq_length, self.seq_length), diagonal=1).int() # (1, seq_length, seq_length)
        decoder_mask = (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & tril_mask # (1, seq_length, seq_length)
        return encoder_mask, decoder_mask

    def __getitem__(self, index):
        src_text = self.dataset[index]['translation'][self.src_lang]
        target_text = self.dataset[index]['translation'][self.target_lang]


        encode_input_tokens = self.src_tokenizer.encode(src_text).ids
        decode_input_tokens = self.target_tokenizer.encode(target_text).ids

        encode_num_paddings = self.seq_length - len(encode_input_tokens) - 2 # exclude <SOS> & <EOS>
        decode_num_paddings = self.seq_length - len(decode_input_tokens) - 1 # exclude <SOS> | <EOS>

        assert encode_num_paddings >= 0, "The length of the source text is too long"
        assert decode_num_paddings >= 0, "The length of the target text is too long"

        encoder_input = torch.cat([
            self.sos_token,
            Tensor(encode_input_tokens).to(torch.int64),
            self.eos_token,
            self.pad_token.repeat(encode_num_paddings).to(torch.int64),
        ])
        decoder_input = torch.cat([
            self.sos_token,
            Tensor(decode_input_tokens).to(torch.int64),
            self.pad_token.repeat(decode_num_paddings).to(torch.int64),
        ])
        label = torch.cat([
            Tensor(decode_input_tokens).to(torch.int64),
            self.eos_token,
            self.pad_token.repeat(decode_num_paddings).to(torch.int64),
        ])
        assert encoder_input.size(0) == self.seq_length
        assert decoder_input.size(0) == self.seq_length
        assert label.size(0) == self.seq_length
        encoder_mask, decoder_mask = self.create_masks(encoder_input, decoder_input)

        return {
            'src_text': src_text,
            'target_text': target_text,
            'encoder_input': encoder_input,
            'decoder_input': decoder_input,
            'encoder_mask': encoder_mask,
            'decoder_mask': decoder_mask,
            'label': label,

        }
