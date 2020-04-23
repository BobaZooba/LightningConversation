import youtokentome as yttm
from typing import List, Optional, Tuple


class Tokenizer:

    SPECIAL_START_SYMBOL = '▁'

    def __init__(self,
                 tokenizer_path: str,
                 need_bos: bool = True,
                 need_eos: bool = True,
                 pad_token: str = '<PAD>',
                 unk_token: str = '<UNK>',
                 bos_token: str = '<BOS>',
                 eos_token: str = '<EOS>',
                 sep_token: str = '<SEP>',
                 context_token: Optional[str] = '<CTX>'):

        self.bpe = yttm.BPE(model=tokenizer_path)
        self.vocab = self.bpe.vocab()

        self.output_type = yttm.OutputType.ID

        self.need_bos = need_bos
        self.need_eos = need_eos

        self.pad_token = pad_token
        self.pad_index = self.vocab.index(self.pad_token)

        self.unk_token = unk_token
        self.unk_index = self.vocab.index(self.unk_token)

        self.bos_token = bos_token
        self.bos_index = self.vocab.index(self.bos_token)

        self.eos_token = eos_token
        self.eos_index = self.vocab.index(self.eos_token)

        self.sep_token = self.SPECIAL_START_SYMBOL + sep_token
        self.sep_index = self.vocab.index(self.sep_token)

        if context_token is not None:
            self.context_token = self.SPECIAL_START_SYMBOL + context_token
            if self.context_token not in self.vocab:
                self.context_token = None
                self.context_index = None
            else:
                self.context_index = self.vocab.index(self.context_token)
        else:
            self.context_token = None
            self.context_index = None

    def __len__(self) -> int:
        return self.bpe.vocab_size()

    def __getitem__(self, index: int) -> str:
        return self.vocab[index]

    def tokenize(self, batch: List[str]) -> List[List[int]]:

        batch = self.bpe.encode(sentences=batch,
                                output_type=self.output_type,
                                bos=self.need_bos,
                                eos=self.need_eos)

        return batch

    def __call__(self, batch: List[str]) -> List[List[int]]:
        return self.tokenize(batch=batch)
