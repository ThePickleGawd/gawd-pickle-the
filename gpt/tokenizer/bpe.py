from collections.abc import Iterable
import os
import io
import regex as re
from typing import Iterator

from gpt.tokenizer.pretokenization import get_pretokens_list_encode, get_pretokens_map_training
from gpt.tokenizer.util import get_merged_pretoken_map_training, get_merged_pretoken_list_encode, init_vocab, PRETOKEN_REGEX


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] = []
    ):
        self.int2byte = vocab
        self.byte2int = {val: key for key, val in vocab.items()}
        self.merges = merges
        self.special_tokens = special_tokens

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] = []
    ):
        pass

    def export(vocab_filepath: str, merges_filepath: str):

        pass

    def encode(self, text: str) -> list[int]:
        # 1. Pretokenize

        # Convert to bytes and wrap like a file
        buffer = io.BytesIO()
        buffer.write(text.encode("utf-8"))
        
        pretoken_list = get_pretokens_list_encode(buffer, self.special_tokens)

        # 2. Apply merges in order
        for merge in self.merges:
            pretoken_list = get_merged_pretoken_list_encode(pretoken_list, merge)

        # 3. Flatten list[tuple[bytes, ...]] to list[int]
        return [self.byte2int[b] for pretoken_tuple in pretoken_list for b in pretoken_tuple]

    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        buffer = ""

        # Don't process a possible special token
        max_special_token_len = max(len(x) for x in  self.special_tokens) if self.special_tokens else 0

        for chunk in iterable:
            buffer += chunk

            if len(buffer) < max_special_token_len:
                continue

            # Don't consider end that could create a special token
            cutoff = max(0, len(buffer) - max_special_token_len)
            safe_text = buffer[:cutoff]
            remainder = buffer[cutoff:]

            # Split into pretokens, don't process the last one; it could be a special token
            last = None
            for match in re.finditer(PRETOKEN_REGEX, safe_text):
                last = match

            if not last:
                continue

            safe_end = last.start()
            yield from self.encode(safe_text[:safe_end])

            buffer = safe_text[safe_end:] + remainder


        if buffer:
            yield from self.encode(buffer)
        
    
    def decode(self, ids: list[int]) -> str:
        # Convert ids to bytes
        text_bytes = b"".join([self.int2byte[id] for id in ids])

        # print([self.int2byte[id].decode("utf-8") for id in ids])

        return text_bytes.decode("utf-8", errors="replace")

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    vocab = init_vocab(special_tokens)
    
    with open(input_path, "rb") as f:
        pretoken_map = get_pretokens_map_training(f, special_tokens)

    merges: list[tuple[bytes, bytes]] = []

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
        pretoken_map = get_merged_pretoken_map_training(pretoken_map, most_freq_pair)
        merges.append(most_freq_pair)
    
    return vocab, merges

if __name__ == "__main__":
    special_tokens = ["<|endoftext|>"]
    vocab, merges = train_bpe("tests/fixtures/tinystories_sample.txt", 1024, special_tokens)
    
    tokenizer = Tokenizer(vocab, merges, special_tokens)
    # print(vocab, merges)

    query = "Hello, this is some text that could be; tokenized?? <|endoftext|> And this is a new doc... lol!"
    print(query)
    ids = tokenizer.encode(query)
    print(ids)
    output = tokenizer.decode(ids)
    print(output)

    assert query == output
    print("\n\nMatches :)")

    print("Encode iterable")
    ids_iterable = []
    for id in tokenizer.encode_iterable(query):
        # print(f"{id}")
        ids_iterable.append(id)
    print(tokenizer.decode(ids_iterable))

    