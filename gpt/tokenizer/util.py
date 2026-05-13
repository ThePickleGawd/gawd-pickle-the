# Return special tokens + all 256 byte encodings
def init_vocab(special_tokens) -> dict[int, bytes]:
    vocab_list = [bytes([i]) for i in range(256)] + [token.encode("utf-8") for token in special_tokens] 
    return {i: vocab_list[i] for i in range(len(vocab_list))}


def get_merged_pretoken_map(pretoken_map: dict[tuple[bytes, ...], int], merge_pair: tuple[bytes, bytes]):
    updated_map = {}

    for pretoken, cnt in pretoken_map.items():
        updated_pretoken = []
        i = 0
        while i < len(pretoken):
            # Add merge token only if it's the most freq pair
            if i + 1 < len(pretoken) and (pretoken[i], pretoken[i+1]) == merge_pair:
                updated_pretoken.append(b"".join(merge_pair))
                i += 2
            else:
                updated_pretoken.append(pretoken[i])
                i += 1
        updated_pretoken = tuple(updated_pretoken)
        updated_map[updated_pretoken] = updated_map.get(updated_pretoken, 0) + cnt

    return updated_map


def get_merged_pretoken_list(pretoken_list: list[tuple[bytes, ...]], merge_pair: tuple[bytes, bytes]):
    updated_list = []

    i = 0
    while i < len(pretoken_list):
        if i + 1 < len(pretoken_list) and (pretoken_list[i], pretoken_list[i+1]) == merge_pair:
            updated_list.append(b"".join(merge_pair))
            i += 2
        else:
            updated_list.append(pretoken_list[i])
            i += 1

    return updated_list
