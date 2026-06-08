from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import os
from typing import BinaryIO
import regex as re
from gpt.tokenizer.util import BYTE_TOKENS, PRETOKEN_REGEX

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def _count_pretokens_chunk(
    args: tuple[str, int, int, tuple[str, ...]],
) -> Counter[tuple[bytes, ...]]:
    input_path, start, end, special_tokens = args

    with open(input_path, "rb") as file:
        file.seek(start)
        chunk = file.read(end - start).decode("utf-8", errors="ignore")

    pretoken_cnts: Counter[tuple[bytes, ...]] = Counter()

    if special_tokens:
        # Regex with largest first to prevent any prefix special tokens firing first
        special_tokens_regex = "|".join(
            re.escape(token)
            for token in sorted(special_tokens, key=lambda x: len(x), reverse=True)
        )
        pretoken_chunks = re.split(special_tokens_regex, chunk)
    else:
        pretoken_chunks = [chunk]

    # Split special tokens, then by pretoken regex
    for pretoken_chunk in pretoken_chunks:
        for pretoken_match in re.finditer(PRETOKEN_REGEX, pretoken_chunk):
            encoded = pretoken_match.group().encode("utf-8")
            pretoken = tuple(BYTE_TOKENS[b] for b in encoded)
            pretoken_cnts[pretoken] += 1

    return pretoken_cnts

# For BPE training, we need dictionary of { pretoken: count }
# We will filter out all special tokens, since we don't need stats on this
def get_pretokens_map_training(
    file: BinaryIO,
    special_tokens: list[str] = [],
    num_processes: int = 8,
) -> dict[tuple[bytes, ...], int]:
    if special_tokens:
        boundaries = find_chunk_boundaries(
            file, num_processes, special_tokens[0].encode("utf-8")
        )
    else:
        file.seek(0, os.SEEK_END)
        boundaries = [0, file.tell()]
        file.seek(0)

    chunks = list(zip(boundaries[:-1], boundaries[1:]))

    jobs = [
        (file.name, start, end, tuple(special_tokens))
        for start, end in chunks
    ]

    pretoken_cnts: Counter[tuple[bytes, ...]] = Counter()

    if num_processes <= 1 or len(jobs) <= 1:
        for job in jobs:
            pretoken_cnts.update(_count_pretokens_chunk(job))
    else:
        with ProcessPoolExecutor(max_workers=num_processes) as pool:
            for chunk_counts in pool.map(_count_pretokens_chunk, jobs):
                pretoken_cnts.update(chunk_counts)

    return dict(pretoken_cnts)


# For Tokenizer encoding, we need a list of the pretokens -- each is a tuple
# [(b"h", b"e", b"l", b"l", b"o"), (b"t", b"h", b"e", b"r", b"e"), ("<|endoftext|>",)]
def get_pretokens_list_encode(file: BinaryIO, special_tokens: list[str] = []) -> list[tuple[bytes, ...]]:

    file.seek(0)
    chunk = file.read().decode("utf-8", errors="ignore")
    pretoken_regex = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    # 1. Build initial pass or pretoken list, seperating special tokens: ["some text", "<|endoftext|>", "long corpus here"]
    pretoken_list_str: list[str] = []
    if special_tokens:
        special_tokens_regex = "|".join(re.escape(token) for token in sorted(special_tokens, key=lambda x: len(x), reverse=True))
        idx = 0
        for special_token_match in re.finditer(special_tokens_regex, chunk):
            # Append non special tokens up to this match
            prev_tokens = chunk[idx:special_token_match.start()]
            if prev_tokens:
                pretoken_list_str.append(prev_tokens)

            # Append the special token
            pretoken_list_str.append(special_token_match.group())
            idx = special_token_match.end()

        if idx != len(chunk):
            pretoken_list_str.append(chunk[idx:])
    else:
        pretoken_list_str = [chunk]
    
    # 2. Within each item which is not a special token, apply the pretokenization regex
    pretoken_list_bytes: list[tuple[bytes, ...]] = []
    for pretoken_chunk in pretoken_list_str:
        if pretoken_chunk in special_tokens:
            pretoken_list_bytes.append((pretoken_chunk.encode("utf-8"),))
            continue

        for pretoken_match in re.finditer(pretoken_regex, pretoken_chunk):
            encoded = pretoken_match.group().encode("utf-8") # bytes: b"abc"
            pretoken_list_bytes.append(tuple(BYTE_TOKENS[b] for b in encoded))
    
    return pretoken_list_bytes
    
## Usage
if __name__ == "__main__":
    with open("tests/fixtures/tinystories_sample.txt", "rb") as f:
        a = get_pretokens_map_training(f)
        b = get_pretokens_map_training(f, ["<|endoftext|>"])
        print(a, b)
