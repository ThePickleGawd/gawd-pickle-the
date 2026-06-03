from collections.abc import Iterable
import os
import io
import pathlib
import regex as re
import json
from typing import Iterator

from gpt.tokenizer.pretokenization import (
    get_pretokens_list_encode,
    get_pretokens_map_training,
)
from gpt.tokenizer.util import (
    get_merged_pretoken_map_training,
    get_merged_pretoken_list_encode,
    gpt2_bytes_to_unicode,
    init_vocab,
    PRETOKEN_REGEX,
)


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] = [],
    ):
        self.int2byte = vocab
        self.byte2int = {val: key for key, val in vocab.items()}
        self.merges = merges
        self.special_tokens = special_tokens

    @classmethod
    def from_files(
        cls, vocab_path: str, merges_path: str, special_tokens: list[str] = []
    ):
        gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
        with open(vocab_path) as vocab_f:
            gpt2_vocab = json.load(vocab_f)
        gpt2_bpe_merges = []
        with open(merges_path) as f:
            for line in f:
                cleaned_line = line.rstrip()
                if cleaned_line and len(cleaned_line.split(" ")) == 2:
                    gpt2_bpe_merges.append(tuple(cleaned_line.split(" ")))
        # The GPT-2 tokenizer uses a remapped unicode encoding for bytes. Let's
        # just return the original bytes, so we don't force students to use
        # any particular encoding scheme.
        vocab = {
            gpt2_vocab_index: bytes(
                [gpt2_byte_decoder[token] for token in gpt2_vocab_item]
            )
            for gpt2_vocab_item, gpt2_vocab_index in gpt2_vocab.items()
        }
        # If any of the special tokens don't exist in the vocab, append them to the vocab.
        if special_tokens:
            for special_token in special_tokens:
                byte_encoded_special_token = special_token.encode("utf-8")
                if byte_encoded_special_token not in set(vocab.values()):
                    vocab[len(vocab)] = byte_encoded_special_token

        merges = [
            (
                bytes([gpt2_byte_decoder[token] for token in merge_token_1]),
                bytes([gpt2_byte_decoder[token] for token in merge_token_2]),
            )
            for merge_token_1, merge_token_2 in gpt2_bpe_merges
        ]
        return cls(vocab, merges, special_tokens)

    def export(self, vocab_path: str, merges_path: str):
        gpt2_byte_encoder = gpt2_bytes_to_unicode()

        # {printable_string_for_token: token_id}
        gpt2_vocab_data = {
            "".join(gpt2_byte_encoder[b] for b in token_bytes): token_id
            for token_bytes, token_id in self.byte2int.items()
        }
        with open(vocab_path, "w") as vocab_f:
            json.dump(gpt2_vocab_data, vocab_f, ensure_ascii=False)

        with open(merges_path, "w", encoding="utf-8") as merges_f:
            merges_f.write("#version: 0.2\n")
            for left, right in self.merges:
                left_str = "".join(gpt2_byte_encoder[b] for b in left)
                right_str = "".join(gpt2_byte_encoder[b] for b in right)
                merges_f.write(f"{left_str} {right_str}\n")

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
        return [
            self.byte2int[b] for pretoken_tuple in pretoken_list for b in pretoken_tuple
        ]

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        buffer = ""

        # Don't process a possible special token
        max_special_token_len = (
            max(len(x) for x in self.special_tokens) if self.special_tokens else 0
        )

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
                byte_pair = (pretoken[i], pretoken[i + 1])
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
    vocab, merges = train_bpe(
        "tests/fixtures/tinystories_sample.txt", 1024, special_tokens
    )

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

    TOKENIZER_PATH = (
        (pathlib.Path(__file__).resolve().parent.parent.parent)
        / "checkpoints"
        / "tokenizer"
    )
    vocab_path = TOKENIZER_PATH / "test_bpe_vocab.json"
    merges_path = TOKENIZER_PATH / "test_bpe_merges.txt"
    tokenizer.export(vocab_path, merges_path)

    new_tokenizer = Tokenizer.from_files(
        vocab_path, merges_path, special_tokens=special_tokens + ["<|newspecialtoken|>"]
    )
    assert output == new_tokenizer.decode(new_tokenizer.encode(query))
    print("new tokenizer success")
