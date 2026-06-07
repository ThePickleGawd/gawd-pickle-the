from functools import lru_cache

PRETOKEN_REGEX = (
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


# Return special tokens + all 256 byte encodings
def init_vocab(special_tokens) -> dict[int, bytes]:
    vocab_list = [bytes([i]) for i in range(256)] + [
        token.encode("utf-8") for token in special_tokens
    ]
    return {i: vocab_list[i] for i in range(len(vocab_list))}


def merge_pretoken(
    pretoken: tuple[bytes, ...], merge_pair: tuple[bytes, bytes]
) -> tuple[bytes, ...]:
    updated_pretoken = []
    i = 0
    while i < len(pretoken):
        # Add merge token only if it's the most freq pair
        if i + 1 < len(pretoken) and (pretoken[i], pretoken[i + 1]) == merge_pair:
            updated_pretoken.append(b"".join(merge_pair))
            i += 2
        else:
            updated_pretoken.append(pretoken[i])
            i += 1
    return tuple(updated_pretoken)


def get_merged_pretoken_list_encode(
    pretoken_list: list[tuple[bytes, ...]], merge_pair: tuple[bytes, bytes]
) -> list[tuple[bytes, ...]]:
    updated_list: list[tuple[bytes]] = []

    for pretoken_tuple in pretoken_list:
        i = 0
        updated_pretoken: list[bytes] = []
        while i < len(pretoken_tuple):
            if (
                i + 1 < len(pretoken_tuple)
                and (pretoken_tuple[i], pretoken_tuple[i + 1]) == merge_pair
            ):
                updated_pretoken.append(b"".join(merge_pair))
                i += 2
            else:
                updated_pretoken.append(pretoken_tuple[i])
                i += 1
        updated_list.append(tuple(updated_pretoken))

    return updated_list


@lru_cache
def gpt2_bytes_to_unicode() -> dict[int, str]:
    """
    Returns a mapping between every possible byte (an integer from 0 to 255) to a
    printable unicode string character representation. This function is taken
    from the GPT-2 code.

    For example, `chr(0)` is `\x00`, which is an unprintable character:

    >>> chr(0)
    '\x00'
    >>> print(chr(0))

    As a result, this function returns a dictionary `d` where `d[0]` returns `Ā`.
    The bytes that are visually printable keep their original string representation [1].
    For example, `chr(33)` returns `!`, and so accordingly `d[33]` returns `!`.
    Note in particular that the space character `chr(32)` becomes `d[32]`, which
    returns 'Ġ'.

    For unprintable characters, the function shifts takes the integer representing
    the Unicode code point of that character (returned by the Python `ord`) function
    and shifts it by 256. For example, `ord(" ")` returns `32`, so the the space character
    ' ' is shifted to `256 + 32`. Since `chr(256 + 32)` returns `Ġ`, we use that as the
    string representation of the space.

    This function can simplify the BPE implementation and makes it slightly easier to
    manually inspect the generated merges after they're serialized to a file.
    """
    # These 188 integers can used as-is, since they are not whitespace or control characters.
    # See https://www.ssec.wisc.edu/~tomw/java/unicode.html.
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    # now get the representations of the other 68 integers that do need shifting
    # each will get mapped chr(256 + n), where n will grow from 0...67 in the loop
    # Get printable representations of the remaining integers 68 integers.
    n = 0
    for b in range(2**8):
        if b not in bs:
            # If this integer isn't in our list of visually-representable
            # charcters, then map it to the next nice character (offset by 256)
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    characters = [chr(n) for n in cs]
    d = dict(zip(bs, characters))
    return d
