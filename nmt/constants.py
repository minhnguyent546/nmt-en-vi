
class SpecialToken:
    SOS = '<SOS>'
    EOS = '<EOS>'
    PAD = '<PAD>'
    UNK = '<UNK>'
    BPE_SUFFIX = '##'
    WORD_PIECE_PREFIX = '##'

class TokenizerModel:
    WORD_LEVEL = 'word_level'
    BPE = 'bpe'
    WORD_PIECE = 'word_piece'

class DatasetName:
    IWSLT2015_EN_VI = 'iwslt2015-en-vi'
    CLANG8 = 'clang8'

class Epoch:
    EPOCH = 'epoch'
    LATEST = 'latest'

class Config:
    LOWERCASE = 'lowercase'
    CONTRACTIONS = 'contractions'
    VI_WORD_SEGMENTTATION = 'vi_word_segmentation'
    REMOVE_UNDERSCORES = 'remove_underscores'


# it's like a underscore, but will not be effected by tokenizer of sacrebleu
# when working with Vietnamese (underscore is used in word segmentation, e.g. underthesea, pyvi)
LOWER_ONE_EIGHTH_BLOCK = u'\u2581'  # "▁"
