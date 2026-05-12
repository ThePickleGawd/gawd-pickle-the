import os
from pretokenization import get_pretoken_map


def init_vocab():
    return [bytes([i]) for i in range(256)]


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    vocab = init_vocab()

    pretokens = get_pretoken_map()
    