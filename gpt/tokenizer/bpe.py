import os
import io
from gpt.tokenizer.pretokenization import get_pretokens, get_pretokens_from_path
from gpt.tokenizer.util import get_merged_pretoken_map, init_vocab


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] = []
    ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens

    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] = []
    ):
        pass

    def encode(self, text: str) -> list[int]:
        # 1. Pretokenize

        # Convert to bytes and wrap like a file
        buffer = io.BytesIO()
        buffer.write(text.encode("utf-8"))
        
        pretoken_map = get_pretokens(buffer)

        # 2. Apply merges in order
        for merge in self.merges:
            pretoken_map = get_merged_pretoken_map(pretoken_map, merge)

        # 3. Output to list
        
    
    def decode(self, ids: list[int]) -> str:
        pass

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    vocab = init_vocab(special_tokens)
    pretoken_map = get_pretokens_from_path(input_path, special_tokens)
    merges: list[tuple[bytes, bytes]] = [] # book keeping (I think for unit test)

    # Merge + add to vocab until vocab size is met
    while len(vocab) < vocab_size:
        # Find most frequent pair
        freq: dict[tuple[bytes, bytes], int] = {}
        for pretoken, cnt in pretoken_map.items():
            for i in range(len(pretoken) - 1):
                byte_pair = (pretoken[i], pretoken[i+1])
                freq[byte_pair] = freq.get(byte_pair, 0) + cnt
        
        if len(freq) == 0:
            break
        most_freq_pair = max(freq, key=freq.get)
        vocab[len(vocab)] = most_freq_pair[0] + most_freq_pair[1]

        # Merge all instances in the pretokens
        pretoken_map = get_merged_pretoken_map(pretoken_map, most_freq_pair)
        merges.append(most_freq_pair)
    
    return vocab, merges

if __name__ == "__main__":
    vocab, merges = train_bpe("tests/fixtures/tinystories_sample.txt", 1024, ["<|endoftext|>"])
    
    tokenizer = Tokenizer(vocab, merges)
    ids = tokenizer.encode("Hello, this is some text that could be; tokenized?? <|endoftext|> And this is a new doc... lol!")
    print(ids)