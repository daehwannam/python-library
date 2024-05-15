
import functools
from typing import List, Callable

import torch
from transformers import LogitsProcessor
from transformers.file_utils import add_start_docstrings
from transformers.generation_logits_process import LOGITS_PROCESSOR_INPUTS_DOCSTRING

from ..iteration import rcopy
from ..decoration import deprecated

from ..torchlib.dnn import mask_tensor, masked_log_softmax


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
    token_ids = rcopy(tokens, coll_fn=tokenizer.convert_tokens_to_ids)

    return tokenizer.decode(
        token_ids,
        skip_special_tokens=skip_special_tokens,
        clean_up_tokenization_spaces=clean_up_tokenization_spaces,
        **kwargs)


def logit_rescaling(logits_processor: LogitsProcessor, num_beams=None, postprocessing_nan=False):
    # if num_beams is not None:
    #     assert not hasattr(logits_processor, '_num_beams')
    #     _num_beams = num_beams
    # elif hasattr(logits_processor, '_num_beams'):
    #     _num_beams = getattr(logits_processor, '_num_beams')
    # else:
    #     raise Exception('`num_beams` should be specified or inferenced from `logits_processor`')

    @functools.wraps(logits_processor)
    def new_logits_processor(*args, **kwargs):
        logits = logits_processor(*args, **kwargs)
        # num_classes = logits.size()[-1]
        # new_logits = torch.nn.functional.log_softmax(logits.view(-1, _num_beams, num_classes), dim=-1)
        new_logits = torch.nn.functional.log_softmax(logits, dim=-1)

        if postprocessing_nan:
            # In `masked_log_softmax`, if all candidate scores are masked by setting the values as -inf,
            # the output of `masked_log_softmax` includes nan values.
            # This is problematic for `GenerationMixin.beam_search`.
            # To fix the problem, the nan values are replaced as -inf.
            new_logits[new_logits.isnan()] = float('-inf')
        return new_logits

    return new_logits_processor


@deprecated
class MaskedLogitsProcessor(LogitsProcessor):
    r"""
    [`LogitsProcessor`] that enforces constrained generation with masking.

    Args:
        prefix_to_mask_fn: (`Callable[[int, torch.Tensor], List[int]]`):
            This function takes a prefix token id sequence and generates a mask tensor
            that constrains candidate token ids in the next step. This function takes 2
            arguments `inputs_ids` and the batch ID `batch_id`. The mask has two values,
            0 and 1. Token ids with 0 values should get penalties and token ids with 1 values
            get no penalties.
    """

    def __init__(self, prefix_to_mask_fn: Callable[[int, torch.Tensor], List[int]], num_beams: int, renormalizing: bool):
        self._prefix_to_mask_fn = prefix_to_mask_fn
        self._num_beams = num_beams
        self.renormalizing = renormalizing

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # input_ids is a shape of (batch_size * num_beams, sequence_length)
        masks = []
        for batch_id, beam_sent in enumerate(input_ids.view(-1, self._num_beams, input_ids.shape[-1])):
            for beam_id, sent in enumerate(beam_sent):
                masks.append(self._prefix_to_mask_fn(batch_id, sent))

        stacked_mask = torch.stack(masks, dim=0).to(input_ids.dtype)
        if self.renormalizing:
            log_probs = masked_log_softmax(scores, mask=stacked_mask, dim=-1)
        else:
            log_probs = mask_tensor(scores, mask=stacked_mask, value=float('-inf'))

        return log_probs
