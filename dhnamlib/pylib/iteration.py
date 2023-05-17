
import itertools
from .function import identity


def unique(seq):
    '''
    >>> unique([10])
    10
    >>> unique([10, 20])
    Traceback (most recent call last):
        ...
    Exception: the length of sequence is longer than 1
    >>> unique([0])
    0
    >>> unique([])
    Traceback (most recent call last):
        ...
    Exception: the length of sequence is 0
    '''
    count = 0
    for elem in seq:
        count += 1
        if count > 1:
            raise Exception("the length of sequence is longer than 1")
    else:
        if count == 1:
            return elem
        else:
            raise Exception("the length of sequence is 0")


def distinct_values(values):
    '''
    >>> set([1, 2, 3, 1])
    {1, 2, 3}
    >>> set(distinct_values([1, 2, 3, 1]))
    Traceback (most recent call last):
        ...
    AssertionError: value "1" is duplicated
    '''
    s = set()
    for value in values:
        assert value not in s, f'value "{value}" is duplicated'
        s.add(value)
        yield value


def distinct_pairs(pairs, **kwargs):
    '''
    >>> dict([['a', 10], ['a', 20]], a=30, b=40)
    {'a': 30, 'b': 40}
    >>> dict(distinct_pairs([['a', 10], ['a', 20]], a=30, b=40))
    Traceback (most recent call last):
        ...
    AssertionError: key "a" is duplicated

    '''
    keys = set()
    for k, v in itertools.chain(pairs, kwargs.items()):
        assert k not in keys, f'key "{k}" is duplicated'
        keys.add(k)
        yield k, v


def dicts2pairs(*args):
    '''
    Dictionaries to key-sequence pairs.

    Example:

    >>> dict(dicts2pairs(dict(a=1, b=2), dict(a=10, b=20), dict(a=100, b=200)))
    {'a': (1, 10, 100), 'b': (2, 20, 200)}
    '''
    if not args:
        yield from ()
    else:
        keys = args[0].keys()
        key_set = set(keys)
        assert all(key_set == set(d.keys()) for d in args)
        for k in keys:
            yield (k, tuple(d[k] for d in args))


def pairs2dicts(**kargs):
    '''
    Key-sequence pairs to dictionaries.

    Example:

    >>> list(pairs2dicts(**{'a': (1, 10, 100), 'b': (2, 20, 200)}))
    [{'a': 1, 'b': 2}, {'a': 10, 'b': 20}, {'a': 100, 'b': 200}]
    '''

    if not kargs:
        yield from ()
    else:
        value = next(iter(kargs.values()))
        # if not all(len(v) == len(value) for v in kargs.values()):
        #     breakpoint()
        assert all(len(v) == len(value) for v in kargs.values())
        for idx in range(len(value)):
            yield {k: v[idx] for k, v in kargs.items()}


# def dzip(*args, **kargs):
#     if args:  # non-empty tuple
#         keys = args[0].keys()
#         key_set = set(keys)
#         assert all(key_set == set(d.keys()) for d in args)
#         for k in keys:
#             yield (k, tuple(d[k] for d in args))

#     elif kargs:  # non-empty dict
#         value = next(iter(kargs.values()))
#         assert all(len(v) == len(value) for v in kargs.values())
#         for idx in range(len(value)):
#             yield {k: v[idx] for k, v in kargs.items()}

#     else:
#         yield from ()  # https://stackoverflow.com/a/36863998/6710003


# def dzip_test():
#     dict_list = [
#         {'a': 1, 'b': 2, 'c': 3},
#         {'a': 4, 'b': 5, 'c': 6},
#         {'a': 7, 'b': 8, 'c': 9}]
#     print(dict_list)

#     merged_dict = dict(dzip(*dict_list))
#     print(merged_dict)

#     dict_list2 = list(dzip(**merged_dict))
#     print(dict_list2)

#     print(dict_list == dict_list2)


def merge_pairs(pairs, merge_fn=None):
    '''
    Example:

    >>> merge_pairs([['a', 1], ['b', 2], ['a', 10], ['c', 30], ['b', 200], ['c', 300]])
    {'a': [1, 10], 'b': [2, 200], 'c': [30, 300]}
    
    >>> merge_pairs([['a', 1], ['b', 2], ['a', 10], ['c', 30], ['b', 200], ['c', 300]], merge_fn=set)
    {'a': {1, 10}, 'b': {200, 2}, 'c': {300, 30}}
    '''

    merged_dict = {}
    for k, v in pairs:
        merged_dict.setdefault(k, []).append(v)
    if merge_fn is not None:
        for k, v in merged_dict.items():
            merged_dict[k] = merge_fn(v)
    return merged_dict


def merge_dicts(dicts, merge_fn=None):
    '''
    Example:

    >>> merge_dicts([dict(a=1, b=2), dict(a=10, c=30), dict(b=200, c=300)])
    {'a': [1, 10], 'b': [2, 200], 'c': [30, 300]}

    >>> merge_dicts([dict(a=1, b=2), dict(a=10, c=30), dict(b=200, c=300)], merge_fn=set)
    {'a': {1, 10}, 'b': {200, 2}, 'c': {300, 30}}
    '''

    return merge_pairs(
        ([k, v] for dic in dicts for k, v in dic.items()),
        merge_fn=merge_fn)


def all_same(seq):
    first = True
    for item in seq:
        if first:
            first_item = item
            first = False
        else:
            if first_item != item:
                return False
    else:
        return True


def keys2values(coll, keys):
    return map(coll.__getitem__, keys)


def get_values_from_pairs(attr_value_pairs, attr_list, key=identity, defaultvalue=None, defaultfunc=None, no_default=False):
    assert sum(arg is not None for arg in [defaultvalue, defaultfunc]) <= 1

    attr_set = set(attr_list)
    assert len(attr_list) == len(attr_set), "Attributes should not overlap."

    attr_key_index_dict = dict(map(reversed, enumerate(attr_list)))
    value_list = [defaultvalue] * len(attr_list)

    for (pair_attr, pair_value) in attr_value_pairs:
        if not attr_set:
            break  # when all values are found

        pair_attr_key = key(pair_attr)
        if pair_attr_key in attr_set:
            attr_set.remove(pair_attr_key)
            value_list[attr_key_index_dict[pair_attr_key]] = pair_value

    if len(attr_set) > 0:
        if no_default:
            for attr in attr_set:
                raise Exception("Cannot find the correspodning value with {}".format(repr(attr)))
        elif defaultfunc is not None:
            for attr in attr_set:
                value_list[attr_key_index_dict[attr]] = defaultfunc()

    return value_list


NO_VALUE = object()
FIND_FAIL_MESSAGE_FORMAT = 'the target value "{target}" cannot be found'


def _iter_idx_and_elem(seq, target, key=identity, default=NO_VALUE, test=None):
    found = False
    if test is None:
        for idx, elem in enumerate(seq):
            if key(elem) == target:
                yield idx, elem
                found = True
        if not found:
            if default is NO_VALUE:
                raise Exception(FIND_FAIL_MESSAGE_FORMAT.format(target=target))
            else:
                yield default, default
    else:
        assert key is identity

        for idx, elem in enumerate(seq):
            if test(elem, target):
                yield idx, elem
                found = True
        if not found:
            if default is NO_VALUE:
                raise Exception(FIND_FAIL_MESSAGE_FORMAT.format(target=target))
            else:
                yield default, default


def find(seq, target, key=identity, default=NO_VALUE, test=None):
    '''
    >>> find(['a', 'b', 'c', 'd'], 'c')
    'c'
    >>> find(['a', 'b', 'c', 'd'], 'C', default='')
    ''
    >>> find(['a', 'b', 'c', 'd'], 'C', key=lambda elem: elem.upper())
    'c'
    >>> find(['a', 'b', 'c', 'd'], 'C', test=lambda elem, target: elem.upper() == target)
    'c'
    '''
    idx, elem = next(_iter_idx_and_elem(seq, target, key=key, default=default, test=test))
    return elem


def index(seq, target, key=identity, default=NO_VALUE, test=None):
    idx, elem = next(_iter_idx_and_elem(seq, target, key=key, default=default, test=test))
    return idx


def finditer(seq, target, key=identity, default=NO_VALUE, test=None):
    for idx, elem in _iter_idx_and_elem(seq, target, key=key, default=default, test=test):
        yield elem


def indexiter(seq, target, key=identity, default=NO_VALUE, test=None):
    for idx, elem in _iter_idx_and_elem(seq, target, key=key, default=default, test=test):
        yield idx


def any_value(seq, is_valid=bool):
    for elem in seq:
        if is_valid(elem):
            return elem
    else:
        return elem  # last value


def is_not_none(value):
    return value is not None


def any_not_none(seq):
    return any_value(seq, is_not_none)


def is_iterable(it):
    # https://stackoverflow.com/questions/1952464/in-python-how-do-i-determine-if-an-object-is-iterable
    try:
        (x for x in it)
    except TypeError:
        return False
    return True


class iterate:
    EMPTY = object()

    def __init__(self, coll):
        self.iterator = iter(coll)
        self.reserved = self.EMPTY

    def __next__(self):
        if self.reserved is self.EMPTY:
            return next(self.iterator)
        else:
            reserved = self.reserved
            self.reserved = self.EMPTY
            return reserved

    def __bool__(self):
        if self.reserved is self.EMPTY:
            try:
                self.reserved = next(self.iterator)
                return True
            except StopIteration:
                return False
        else:
            return True


def erange(*args):
    'extended range'

    start = 0
    step = 1

    if len(args) == 1:
        [stop] = args
    elif len(args) == 2:
        start, stop = args
    else:
        assert len(args) == 3
        start, stop, step = args

    if stop == float('inf'):
        return itertools.count(start, step)
    else:
        return range(start, stop, step)


def nest(first, *others):
    if others:
        for first_item in first:
            for other_items in nest(*others):
                yield (first_item,) + other_items
    else:
        for first_item in first:
            yield (first_item,)


def chunk_sizes(total_num, num_chunks):
    assert total_num >= num_chunks

    if total_num % num_chunks > 0:
        max_chunk_size = total_num // num_chunks + 1
        for i in range(num_chunks):
            if i + (max_chunk_size - 1) * num_chunks < total_num:
                yield max_chunk_size
            else:
                yield max_chunk_size - 1
    else:
        yield from itertools.repeat(total_num // num_chunks, num_chunks)


def idxmax(items, *, key):
    assert len(items) > 0

    def pair_key(pair):
        idx, item = pair
        return key(item)

    max_idx, max_item = max(enumerate(items), key=pair_key)
    return max_idx


def idxmin(items, *, key):
    return idxmax(items, key=lambda x: -key(x))


def replace_with_last(items, idx):
    item = items[idx]
    items[idx] = items[-1]
    items.pop()
    return item


def split_by_indices(seq, indices):
    last_idx = 0
    for idx in indices:
        yield seq[last_idx: idx]
        last_idx = idx
    yield seq[last_idx:]


def partition(seq, n, strict=True, fill_value=None):
    # Similar to Hy's partition
    # https://github.com/hylang/hy/blob/0.18.0/hy/core/language.hy

    assert not strict or fill_value is None
    it = iter(seq)
    remaining = True

    def get_next_item():
        nonlocal remaining
        try:
            return next(it)
        except StopIteration:
            remaining = False
            return fill_value

    while remaining:
        first_item = get_next_item()
        if remaining:
            items = [first_item]
            if strict:
                for _ in range(n - 1):
                    item = get_next_item()
                    if remaining:
                        items.append(item)
                    else:
                        raise Exception('The total number of items is not divided by {}'.format(n))
            else:
                items.extend(get_next_item() for _ in range(n - 1))
            yield tuple(items)
        else:
            break


def flatten(coll, coll_type=None):
    # Similar to Hy's flatten
    # https://github.com/hylang/hy/blob/0.18.0/hy/core/language.hy

    if coll_type is None:
        coll_type = type(coll)
    elif isinstance(coll_type, list):
        coll_type = tuple(coll_type)

    flattened = []

    def recurse(obj):
        if isinstance(obj, coll_type):
            for elem in obj:
                recurse(elem)
        else:
            flattened.append(obj)

    recurse(coll)
    return flattened


def chainelems(coll):
    for elem in coll:
        for item in elem:
            yield item
