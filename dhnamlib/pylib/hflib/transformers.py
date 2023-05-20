

def iter_token_ids(tokenizer):
    return range(len(tokenizer))


def iter_tokens(tokenizer):
    for token in map(tokenizer.convert_ids_to_tokens, iter_token_ids(tokenizer)):
        assert token is not None
        yield token


def iter_id_token_pairs(tokenizer):
    token_ids = tuple(iter_token_ids(tokenizer))
    tokens = map(tokenizer.convert_ids_to_tokens, token_ids)
    for token_id, token in zip(token_ids, tokens):
        assert token is not None
        yield token_id, token


def all_default_special_tokens(tokenizer):
    # the output doesn't include added tokens
    return tokenizer.all_special_tokens


def iter_default_non_special_tokens(tokenizer):
    # the output doesn't include added tokens
    default_special_ids = set(tokenizer.default_special_ids)
    for token_id in range(tokenizer.vocab_size):
        if token_id not in default_special_ids:
            special_token = tokenizer.convert_ids_to_tokens(token_id)
            assert special_token is not None
            assert isinstance(special_token, str)
            yield special_token
