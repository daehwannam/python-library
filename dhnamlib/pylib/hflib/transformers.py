
def get_all_tokens(tokenizer):
    return tuple(map(tokenizer.convert_ids_to_tokens, range(tokenizer.vocab_size)))


def get_all_special_tokens(tokenizer):
    return tokenizer.all_special_tokens


def get_all_non_special_tokens(tokenizer):
    all_special_ids = set(tokenizer.all_special_ids)
    all_non_special_tokens = tuple(
        tokenizer.convert_ids_to_tokens(token_id)
        for token_id in range(tokenizer.vocab_size)
        if token_id not in all_special_ids)
    return all_non_special_tokens
