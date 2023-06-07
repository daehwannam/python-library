
from dhnamlib.pylib.iteration import apply_recursively

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
    default_special_ids = set(tokenizer.all_special_ids)
    for token_id in range(tokenizer.vocab_size):
        if token_id not in default_special_ids:
            special_token = tokenizer.convert_ids_to_tokens(token_id)
            assert special_token is not None
            assert isinstance(special_token, str)
            yield special_token


def join_tokens(
        tokenizer,
        tokens,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = True,
        **kwargs):
    token_ids = apply_recursively(tokens, coll_fn=tokenizer.convert_tokens_to_ids)

    return tokenizer.decode(
        token_ids,
        skip_special_tokens=skip_special_tokens,
        clean_up_tokenization_spaces=clean_up_tokenization_spaces,
        **kwargs)
