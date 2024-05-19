import contractions
import html
import re

import torch
from torch import Tensor

from tokenizers import Tokenizer

from nmt.constants import Config, SpecialToken
from nmt.utils import misc as misc_util, model as model_util
from nmt.utils.tokenizer import MosesDetokenizer, MosesTokenizer
from transformer import Transformer


class Translator:
    def __init__(
        self,
        model: Transformer,
        src_tokenizer: Tokenizer,
        target_tokenizer: Tokenizer,
        config: dict | None = None,
    ) -> None:
        self.model = model
        self.src_tokenizer = src_tokenizer
        self.target_tokenizer = target_tokenizer
        self.config = config if config is not None else {}

        assert src_tokenizer.token_to_id(SpecialToken.SOS) == target_tokenizer.token_to_id(SpecialToken.SOS)
        assert src_tokenizer.token_to_id(SpecialToken.EOS) == target_tokenizer.token_to_id(SpecialToken.EOS)
        assert src_tokenizer.token_to_id(SpecialToken.PAD) == target_tokenizer.token_to_id(SpecialToken.PAD)

        self.sos_token_id = src_tokenizer.token_to_id(SpecialToken.SOS)
        self.eos_token_id = src_tokenizer.token_to_id(SpecialToken.EOS)
        self.pad_token_id = src_tokenizer.token_to_id(SpecialToken.PAD)

        self.moses_tokenizer = MosesTokenizer(lang='en')
        self.moses_detokenizer = MosesDetokenizer(lang='vi')

        self.model.eval()

    def __call__(
        self,
        src_text: str,
        beam_size: int = 1,
        beam_return_topk: int = 1,
        max_seq_length: int = 120,
        tokenized: bool = False,
    ) -> str | list[str]:
        src_text = self._preprocess_sentence(src_text, tokenized=tokenized)
        encoder_input = self._make_encoder_input(src_text)
        cand_list = None
        cand_text_list = None

        with torch.no_grad():
            if beam_size is not None and beam_size > 1:
                cand_list = model_util.beam_search_decode(self.model, self.model.device, beam_size,
                                                          encoder_input, self.target_tokenizer,
                                                          max_seq_length, return_topk=beam_return_topk)
                pred_token_ids = cand_list[0]
            else:
                pred_token_ids = model_util.greedy_search_decode(self.model, self.model.device, encoder_input,
                                                                 self.target_tokenizer, max_seq_length)

            pred_text = self.target_tokenizer.decode(pred_token_ids.detach().cpu().numpy(), skip_special_tokens=False)
            if cand_list is not None:
                cand_text_list = [
                    self.target_tokenizer.decode(cand.detach().cpu().numpy(), skip_special_tokens=False)
                    for cand in cand_list
                ]

            if cand_text_list is not None:
                cand_text_list = [self._postprocess_sentence(cand) for cand in cand_text_list]
                return cand_text_list
            else:
                return self._postprocess_sentence(pred_text)

    def _make_encoder_input(self, src_text: str) -> Tensor:
        encode_input_tokens = self.src_tokenizer.encode(src_text).ids
        encoder_input = torch.cat([
            Tensor([self.sos_token_id]),
            Tensor(encode_input_tokens),
            Tensor([self.eos_token_id]),
        ]).type(torch.int32)

        return encoder_input

    def _preprocess_sentence(self, sentence: str, tokenized: bool = False) -> str:
        if not tokenized:
            sentence = self.moses_tokenizer(sentence, format='text')

        sentence = sentence.strip()
        sentence = re.sub(r'\s(&[a-zA-Z]+;)', r'\1', sentence)
        sentence = html.unescape(sentence)

        if misc_util.is_enabled(self.config['preprocess'], Config.LOWERCASE):
            sentence = sentence.lower()
        if misc_util.is_enabled(self.config['preprocess'], Config.CONTRACTIONS):
            sentence = re.sub(r"\s('[\w\d]+)", r'\1', sentence)
            sentence = contractions.fix(sentence)

        return sentence

    def _postprocess_sentence(self, sentence: str) -> str:
        if sentence.startswith(SpecialToken.SOS):
            sentence = sentence[len(SpecialToken.SOS):]
        if sentence.endswith(SpecialToken.EOS):
            sentence = sentence[:-len(SpecialToken.EOS)]
        if misc_util.is_enabled(self.config['postprocess'], Config.REMOVE_UNDERSCORES):
            sentence = sentence.replace('_', ' ')
        sentence = sentence.strip()

        return self.moses_detokenizer(sentence, format='text')
