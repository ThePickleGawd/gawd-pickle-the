import os
from gpt.tokenizer.pretokenization import get_pretoken_map


# Return special tokens + all 256 byte encodings
def init_vocab(special_tokens) -> dict[int, bytes]:
    vocab_list = [bytes([i]) for i in range(256)] + [token.encode("utf-8") for token in special_tokens] 
    return {i: vocab_list[i] for i in range(len(vocab_list))}

def get_merged_pretoken_map(pretoken_map: dict[tuple[bytes, ...], int], most_freq_pair: tuple[bytes, bytes]):
    updated_map = {}

    for pretoken, cnt in pretoken_map.items():
        updated_pretoken = []
        i = 0
        while i < len(pretoken):
            # Add merge token only if it's the most freq pair
            if i + 1 < len(pretoken) and (pretoken[i], pretoken[i+1]) == most_freq_pair:
                updated_pretoken.append(b"".join(most_freq_pair))
                i += 2
            else:
                updated_pretoken.append(pretoken[i])
                i += 1
        updated_pretoken = tuple(updated_pretoken)
        updated_map[updated_pretoken] = updated_map.get(updated_pretoken, 0) + cnt

    return updated_map


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    vocab = init_vocab(special_tokens)
    pretoken_cnts = get_pretoken_map(input_path, special_tokens)
    merges: list[tuple[bytes, bytes]] = [] # book keeping (I think for unit test)

    # Merge + add to vocab until vocab size is met
    while len(vocab) < vocab_size:
        # Find most frequent pair
        freq: dict[tuple[bytes, bytes], int] = {}
        for pretoken, cnt in pretoken_cnts.items():
            for i in range(len(pretoken) - 1):
                byte_pair = (pretoken[i], pretoken[i+1])
                freq[byte_pair] = freq.get(byte_pair, 0) + cnt
        
        if len(freq) == 0:
            break
        most_freq_pair = max(freq, key=freq.get)
        vocab[len(vocab)] = most_freq_pair[0] + most_freq_pair[1]

        # Merge all instances in the pretokens
        pretoken_cnts = get_merged_pretoken_map(pretoken_cnts, most_freq_pair)
        merges.append(most_freq_pair)
    
    return vocab, merges

if __name__ == "__main__":
    vocab, merges = train_bpe("tests/fixtures/tinystories_sample.txt", 1024, ["<|endoftext|>"])
    print(vocab)
    print(merges)
