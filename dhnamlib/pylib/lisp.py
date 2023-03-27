
from itertools import chain

from dhnamlib.pylib.iteration import partition
from dhnamlib.pylib.iteration import split_by_indices


def parse_hy_args(symbols):
    '''
    e.g. parse_hy_args(*'100 200 300 :x 400 :y 500'.split())
    '''
    args = []

    for idx, symbol in enumerate(symbols):
        if symbol.startswith(':'):
            break
        else:
            args.append(symbol)
    else:
        idx += 1

    def remove_colon(k):
        assert k[0] == ':'
        return k[1:]

    pairs = partition(symbols[idx:], 2)
    kwargs = dict((remove_colon(k), v) for k, v in pairs)

    return args, kwargs


def _get_quoted_paren_index_pairs(s, recursive=False):
    # https://stackoverflow.com/a/29992019

    '''
    e.g.
    >>> s = "(a b '(c d '(e f)))"
    >>> pairs = _get_quoted_paren_index_pairs(s)
    >>> pair = pairs[0]
    >>> s[pair[0]: pair[1]]

    "(c d '(e f)"
    '''

    opening_stack = []  # stack of indices of opening parentheses
    pairs = []
    is_last_char_quote = False
    num_quoted_exprs_under_parsing = 0

    for i, c in enumerate(s):
        if c == "'":
            is_last_char_quote = True
        else:
            if c in ['(', '[']:
                opening_stack.append((i, c, is_last_char_quote))
                if is_last_char_quote:
                    num_quoted_exprs_under_parsing += 1
            elif c in [')', ']']:
                try:
                    idx, char, quoted = opening_stack.pop()
                    assert (char == '(' and c == ')') or (char == '[' and c == ']')
                    if quoted:
                        num_quoted_exprs_under_parsing -= 1
                        if recursive or num_quoted_exprs_under_parsing == 0:
                            pairs.append((idx, i))
                except IndexError:
                    print('Too many closing parentheses')
            is_last_char_quote = False
    if opening_stack:  # check if stack is empty afterwards
        print('Too many opening parentheses')

    return pairs


def remove_comments(text):
    splits = text.split('\n')
    # only remove lines that starts with comments
    return '\n'.join(split for split in splits if not split.lstrip().startswith(';'))


def preprocess_quotes(text, **kwargs):
    if not any(kwargs.values()):
        raise Exception('No option is enabled')

    round_to_string = kwargs.get('round_to_string')
    if not isinstance(round_to_string, (tuple, list)):
        round_to_string_prefixes = (round_to_string,)
    else:
        round_to_string_prefixes = sorted(round_to_string, key=lambda x: type(x) != str)

    square_to_round = kwargs.get('square_to_round')

    quoted_paren_index_pairs = _get_quoted_paren_index_pairs(text)
    region_index_pairs = tuple((i - 1, j + 1)for i, j in quoted_paren_index_pairs)
    region_start_indices = set(start for start, end in region_index_pairs)
    region_indices = sorted(set(chain(*region_index_pairs)))
    splits = split_by_indices(text, region_indices)
    regions = []
    for start_idx, split in zip(chain([0], region_indices), splits):
        region = split
        if start_idx in region_start_indices:
            if split.startswith("'("):
                # Round brackets
                # e.g. split == '(some symbols)
                #      region == "(some symbols)"
                for prefix in round_to_string_prefixes:
                    if isinstance(prefix, str) and \
                       (start_idx > 0 and text[start_idx - 1] == prefix):
                        regions[-1] = regions[-1][:-len(prefix)]
                        round_to_string_matched = True
                        break
                    elif prefix is True:
                        round_to_string_matched = True
                        break
                else:
                    round_to_string_matched = False
                if round_to_string_matched:
                    region = '"{}"'.format(split[1:].replace('"', r'\"'))
            else:
                assert split.startswith("'[")
                if square_to_round is not None:
                    # Square brackets
                    # e.g. split == '[some symbols]
                    #      region == '(some symbols)
                    region = split.replace('[', '(').replace(']', ')')
                    # region = "'({})".format(split[2:-1])
        regions.append(region)
    return ''.join(regions)
