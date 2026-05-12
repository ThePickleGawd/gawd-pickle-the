import os
from pretokenization import get_pretoken_map


# Return special tokens + all 256 byte encodings
def init_vocab(special_tokens) -> dict[int, bytes]:
    vocab_list = [bytes(token) for token in special_tokens] + [bytes([i]) for i in range(256)]
    return {i: vocab_list[i] for i in range(len(vocab_list))}

def get_merged_pretoken_map(pretoken_map, most_freq_pair):
    updated_map = {}

    for pretoken, cnt in pretoken_map.items():
        updated_pretoken = bytes()

    return updated_map


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    vocab = init_vocab(special_tokens)
    pretoken_cnts = get_pretoken_map(input_path, special_tokens)
    merges: tuple[bytes, bytes] = [] # book keeping (I think for unit test)

    # Merge + add to vocab until vocab size is met
    while len(vocab.items()) < vocab_size:
        # Find most frequent pair
        freq: dict[tuple[bytes, ...], int] = {}
        for pretoken, cnt in pretoken_cnts.items():
            for i in range(len(pretoken) - 1):
                byte_pair = pretoken[i:i+1]
                freq[byte_pair] = freq[byte_pair].get(0) + cnt
        
        most_freq_pair = max(freq, key=freq.get)
        vocab[len(vocab.items())] = most_freq_pair

        # Merge all instances in the pretokens
        pretoken_cnts = get_merged_pretoken_map(pretoken_cnts, most_freq_pair)
        