import string
from typing import Callable, Dict, Iterable, List, Optional


class Tokeniser:
    def __init__(
        self,
        vocab: Optional[Iterable[str]] = None,
        tokeniser: Callable[[str], List[str]] = lambda input: list(input),
        pad_token: str = "[PAD]",
        unk_token: str = "[UNK]",
        bos_token: str = "[BOS]",
        eos_token: str = "[EOS]",
    ):
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.bos_token = bos_token
        self.eos_token = eos_token

        self.tokens = [
            self.pad_token,
            self.unk_token,
            self.bos_token,
            self.eos_token,
        ]
        self.unk_index = self.tokens.index(self.unk_token)
        self.tokens.extend(vocab or list(string.printable))

        self.vocab: Dict[str, int] = dict(
            (tok, idx) for idx, tok in enumerate(self.tokens)
        )

        self.ids_to_tokens: Dict[int, str] = dict(enumerate(self.tokens))

        self.tokeniser = tokeniser

    def tokenise(self, text: str) -> List[str]:
        """Tokenize given text."""
        return self.tokeniser(text)

    def __call__(self, raw_text: str) -> List[int]:
        """Given raw text, tokenise and return as a list of ids"""
        return self.convert_tokens_to_ids(self.tokenise(raw_text))

    def convert_token_to_id(self, token: str) -> int:
        """Convert a token in an id using the vocab."""
        return self.vocab.get(token, self.unk_index)

    def convert_id_to_token(self, id: int) -> str:
        """Convert an id in a token using the vocab."""
        return self.ids_to_tokens.get(id, self.unk_token)

    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        """Convert list of tokens in list of ids using the vocab."""
        return [self.convert_token_to_id(token) for token in tokens]

    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        """Convert list of ids in list of tokens using the vocab."""
        return [self.convert_id_to_token(id) for id in ids]

    @property
    def vocab_size(self) -> int:
        """Vocabulary size."""
        return len(self.vocab)

    @property
    def pad_token_id(self) -> int:
        """Id of pad_token in the vocab."""
        return self.convert_token_to_id(self.pad_token)

    @property
    def unk_token_id(self) -> int:
        """Id of unk_token in the vocab."""
        return self.convert_token_to_id(self.unk_token)

    @property
    def bos_token_id(self) -> int:
        """Id of bos_token in the vocab."""
        return self.convert_token_to_id(self.bos_token)

    @property
    def eos_token_id(self) -> int:
        """Id of eos_token in the vocab."""
        return self.convert_token_to_id(self.eos_token)
