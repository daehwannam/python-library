
import itertools
import importlib.util

from .function import identity
from .constant import NO_VALUE
from .exception import DuplicateValueError, NotFoundError
from .typeutil import is_type
from .decoration import deprecated


def unique(seq):
    '''
    >>> unique([10])
    10
    >>> unique([10, 20])
    Traceback (most recent call last):
        ...
    dhnamlib.pylib.exception.DuplicateValueError: the length of sequence is longer than 1
    >>> unique([0])
    0
    >>> unique([])
    Traceback (most recent call last):
        ...
    dhnamlib.pylib.exception.DuplicateValueError: the length of sequence is 0
    '''
    count = 0
    for elem in seq:
        count += 1
        if count > 1:
            raise DuplicateValueError("the length of sequence is longer than 1")
    else:
        if count == 1:
            return elem
        else:
            raise DuplicateValueError("the length of sequence is 0")


def checkup(items, *, predicate):
    '''
    >>> tuple(checkup([1, 3, 5, 7, 9, 10], predicate=lambda x: x % 2 == 1))
    Traceback (most recent call last):
        ...
    Exception: 10 does not satisfy the predicate
    '''

    for item in items:
        if predicate(item):
            yield item
        else:
            raise Exception(f'{item} does not satisfy the predicate')

def distinct_values(values):
    '''
    >>> set([1, 2, 3, 1])
    {1, 2, 3}
    >>> set(distinct_values([1, 2, 3, 1]))
    Traceback (most recent call last):
        ...
    dhnamlib.pylib.exception.DuplicateValueError: value "1" is duplicated
    '''
    s = set()
    for value in values:
        if value in s:
            raise DuplicateValueError(f'value "{value}" is duplicated')
        s.add(value)
        yield value


def distinct_pairs(pairs=(), **kwargs):
    '''
    >>> dict([['a', 10], ['a', 20]], a=30, b=40)
    {'a': 30, 'b': 40}
    >>> dict(distinct_pairs([['a', 10], ['a', 20]], a=30, b=40))
    Traceback (most recent call last):
        ...
    dhnamlib.pylib.exception.DuplicateValueError: key "a" is duplicated

    '''
    keys = set()
    for k, v in itertools.chain(pairs, kwargs.items()):
        if k in keys:
            raise DuplicateValueError(f'key "{k}" is duplicated')
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


def pairs2dicts(pairs=(), **kwargs):
    '''
    Key-sequence pairs to dictionaries.

    Example:

    >>> list(pairs2dicts(**{'a': (1, 10, 100), 'b': (2, 20, 200)}))
    [{'a': 1, 'b': 2}, {'a': 10, 'b': 20}, {'a': 100, 'b': 200}]
    >>> list(pairs2dicts(a=(1, 10, 100), b=(2, 20, 200)))
    [{'a': 1, 'b': 2}, {'a': 10, 'b': 20}, {'a': 100, 'b': 200}]
    >>> list(pairs2dicts([['a', (1, 10, 100)], ['b', (2, 20, 200)]]))
    [{'a': 1, 'b': 2}, {'a': 10, 'b': 20}, {'a': 100, 'b': 200}]
    '''

    merged_pairs = tuple(itertools.chain(pairs, kwargs.items()))

    if len(merged_pairs) == 0:
        yield from ()
    else:
        _, first_value = merged_pairs[0]
        assert all(len(value) == len(first_value) for key, value in merged_pairs)
        for idx in range(len(first_value)):
            yield {key: value[idx] for key, value in merged_pairs}


# def dzip(*args, **kwargs):
#     if args:  # non-empty tuple
#         keys = args[0].keys()
#         key_set = set(keys)
#         assert all(key_set == set(d.keys()) for d in args)
#         for k in keys:
#             yield (k, tuple(d[k] for d in args))

#     elif kwargs:  # non-empty dict
#         value = next(iter(kwargs.values()))
#         assert all(len(v) == len(value) for v in kwargs.values())
#         for idx in range(len(value)):
#             yield {k: v[idx] for k, v in kwargs.items()}

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

    >>> merge_dicts([dict(a=1, b=2), dict(c=30, d=40)], merge_fn=unique)  # similar to `distinct_pairs`
    {'a': 1, 'b': 2, 'c': 30, 'd': 40}
    '''

    return merge_pairs(
        ([k, v] for dic in dicts for k, v in dic.items()),
        merge_fn=merge_fn)


def _filter_pair_values(fn, /, pairs, kwargs={}):
    for key, value in itertools.chain(pairs, kwargs.items()):
        if fn(value):
            yield key, value


def filter_dict_values(fn, dic):
    return _filter_pair_values(fn, dic.items())


def not_none_valued_pairs(pairs=(), **kwargs):
    '''
    Example:

    >>> dict(not_none_valued_pairs(a=10, b=None, c=20))
    {'a': 10, 'c': 20}
    '''
    return _filter_pair_values(is_not_none, pairs, kwargs)


def not_none_valued_dict(pairs=(), **kwargs):
    '''
    Example:

    >>> not_none_valued_dict(a=10, b=None, c=20)
    {'a': 10, 'c': 20}
    '''
    if isinstance(pairs, dict):
        _pairs = pairs.items()
    else:
        _pairs = pairs

    return dict(not_none_valued_pairs(_pairs, **kwargs))


COLL_TYPE_STR_ERROR_MESSAGE = 'str type is not allowed as it raises RecursionError due to infinite recursion.'


def rmap(
        fn, coll,
        coll_fn=None, dict_fn=None,
        coll_type=(list, tuple, set), dict_type=dict
):
    '''
    Recursive map.
    Similar to `rmapcar` in the book "On Lisp".

    Example:

    >>> rmap(str, [1, [2, 3], [4, 5, [6, 7, 8]]], coll_fn=tuple)
    ('1', ('2', '3'), ('4', '5', ('6', '7', '8')))

    >>> rmap(str, [{'a': [1, 2, 3], 'b': [4, 5, 6]}, {'c': [7, 8, 9]}], coll_fn=tuple, dict_fn=list)
    ([('a', ('1', '2', '3')), ('b', ('4', '5', '6'))], [('c', ('7', '8', '9'))])
    '''

    assert not isinstance(coll, str), COLL_TYPE_STR_ERROR_MESSAGE
    assert is_type(coll_type)
    assert is_type(dict_type)

    def recurse(obj):
        if isinstance(obj, coll_type):
            _coll_fn = coll_fn or type(obj)
            return _coll_fn(recurse(elem) for elem in obj)
        elif isinstance(obj, dict_type):
            _dict_fn = dict_fn or type(obj)
            return _dict_fn((recurse(k), recurse(v)) for k, v in obj.items())
        else:
            return fn(obj)

    return recurse(coll)


def rcopy(
        obj,
        dict_fn=None, coll_fn=None,
        coll_type=(list, tuple, set), dict_type=dict
):
    '''
    Recursive copy.

    Example:

    >>> rcopy([1, [2, 3], [4, 5, [6, 7, 8]]], coll_fn=list)
    [1, [2, 3], [4, 5, [6, 7, 8]]]

    >>> rcopy([{'a': [1, 2, 3], 'b': [4, 5, 6]}, {'c': [7, 8, 9]}], coll_fn=tuple, dict_fn=list)
    ([('a', (1, 2, 3)), ('b', (4, 5, 6))], [('c', (7, 8, 9))])
    '''

    return rmap(identity, obj,
                coll_fn=coll_fn, dict_fn=dict_fn,
                coll_type=coll_type, dict_type=dict_type)


def rmemberif(predicate, coll,
              coll_type=(list, tuple, set), dict_type=dict,
              default=NO_VALUE):
    '''
    Recursive member-if

    Example:

    >>> rmemberif(lambda x: (sum(flatten(x)) < 0), [1, [-2, 3, -4], 5])
    [-2, 3, -4]

    >>> rmemberif(lambda x: isinstance(x, list), {'x': {'a': 1, 'b': 2, 'c': [3, 4, 5]}})
    [3, 4, 5]
    '''

    assert not isinstance(coll, str), COLL_TYPE_STR_ERROR_MESSAGE
    assert is_type(coll_type)
    assert is_type(dict_type)

    NO_OUTPUT = object()

    def is_output(obj):
        return obj is not NO_OUTPUT

    def recurse(obj):
        if predicate(obj):
            return obj
        elif isinstance(obj, coll_type):
            return any_value(map(recurse, obj), is_valid=is_output, default=NO_OUTPUT)
        elif isinstance(obj, dict_type):
            return any_value(map(recurse, chainelems(obj.items())), is_valid=is_output, default=NO_OUTPUT)
        else:
            return NO_OUTPUT

    output = recurse(coll)
    if output is NO_OUTPUT:
        if default is NO_VALUE:
            raise NotFoundError('No value satisfies the predicate.')
        else:
            return default
    else:
        return output


@deprecated
def rmember(target, coll,
            coll_type=(list, tuple, set), dict_type=dict):
    def predicate(obj):
        return obj == target

    return rmemberif(predicate, coll, coll_type=coll_type, dict_type=dict_type)


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


def keys2items(coll, keys):
    return zip(keys, map(coll.__getitem__, keys))


def get_values_from_pairs(attr_value_pairs, attr_list, key=identity, defaultvalue=None, defaultfunc=None, no_default=False):
    '''
    Example:

    >>> get_values_from_pairs([['a', 10], ['b', 20], ['c', 30]], ['b', 'a'])
    [20, 10]
    '''
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
                raise NotFoundError("Cannot find the correspodning value with {}".format(repr(attr)))
        elif defaultfunc is not None:
            for attr in attr_set:
                value_list[attr_key_index_dict[attr]] = defaultfunc()

    return value_list


FIND_FAIL_MESSAGE_FORMAT = 'the target value "{target}" cannot be found'


def _iter_idx_and_elem(seq, target, key=identity, default=NO_VALUE, test=None, reverse=False):
    _enumerate = reversed_enumerate if reverse else enumerate
    found = False
    if test is None:
        for idx, elem in _enumerate(seq):
            if key(elem) == target:
                yield idx, elem
                found = True
        if not found:
            if default is NO_VALUE:
                raise NotFoundError(FIND_FAIL_MESSAGE_FORMAT.format(target=target))
            else:
                yield default, default
    else:
        assert key is identity

        for idx, elem in _enumerate(seq):
            if test(elem, target):
                yield idx, elem
                found = True
        if not found:
            if default is NO_VALUE:
                raise NotFoundError(FIND_FAIL_MESSAGE_FORMAT.format(target=target))
            else:
                yield default, default


def find(seq, target, key=identity, default=NO_VALUE, test=None, reverse=False):
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
    idx, elem = next(_iter_idx_and_elem(seq, target, key=key, default=default, test=test, reverse=reverse))
    return elem


def index(seq, target, key=identity, default=NO_VALUE, test=None, reverse=False):
    '''
    >>> index(['a', 'b', 'c', 'd', 'e', 'c', 'f'], 'C', test=lambda elem, target: elem.upper() == target)
    2
    >>> index(['a', 'b', 'c', 'd', 'e', 'c', 'f'], 'C', test=lambda elem, target: elem.upper() == target, reverse=True)
    5
    '''
    idx, elem = next(_iter_idx_and_elem(seq, target, key=key, default=default, test=test, reverse=reverse))
    return idx


def finditer(seq, target, key=identity, default=NO_VALUE, test=None, reverse=False):
    for idx, elem in _iter_idx_and_elem(seq, target, key=key, default=default, test=test, reverse=reverse):
        yield elem


def indexiter(seq, target, key=identity, default=NO_VALUE, test=None, reverse=False):
    for idx, elem in _iter_idx_and_elem(seq, target, key=key, default=default, test=test, reverse=reverse):
        yield idx


@deprecated
def findif(seq, predicate, default=NO_VALUE):
    for elem in seq:
        if predicate(elem):
            return elem
    else:
        return default


def any_value(seq, is_valid=bool, default=NO_VALUE):
    for elem in seq:
        if is_valid(elem):
            return elem
    else:
        if default is NO_VALUE:
            raise NotFoundError('no value is satisfied')
        else:
            return default


def is_not_none(value):
    return value is not None


def any_not_none(seq, default=NO_VALUE):
    return any_value(seq, is_valid=is_not_none, default=default)


def is_iterable(it):
    # https://stackoverflow.com/questions/1952464/in-python-how-do-i-determine-if-an-object-is-iterable
    try:
        (x for x in it)
    except TypeError:
        return False
    return True


class iterate:
    '''
    Example:

    >>> iterator  = iterate(range(5))
    >>> total = 0
    >>> while iterator:
    ...     total += next(iterator)
    ...
    >>> total
    10

    >>> iterator  = iterate(range(5))
    >>> print([next(iterator), next(iterator), next(iterator)])
    [0, 1, 2]
    >>> iterator.restore(10)
    >>> print([next(iterator), next(iterator), next(iterator)])
    [10, 3, 4]

    >>> iterator  = iterate(range(5))
    >>> for x in iterator:
    ...     if not iterator:
    ...         print('The last value: ', x)
    The last value:  4
    '''

    def __init__(self, coll):
        self.iterator = iter(coll)
        self._restored_items = []

    def __next__(self):
        if len(self._restored_items) > 0:
            return self._restored_items.pop()
        else:
            return next(self.iterator)

    def __bool__(self):
        if self._restored_items:
            return True
        else:
            try:
                item = next(self.iterator)
                self._restored_items.append(item)
                return True
            except StopIteration:
                return False

    def __iter__(self):
        while self:
            yield next(self)

    def restore(self, item):
        self._restored_items.append(item)


def exrange(*args):
    '''
    Extended range.

    Example:

    >>> tuple(zip('abcdefg', exrange(1, float('inf'))))
    (('a', 1), ('b', 2), ('c', 3), ('d', 4), ('e', 5), ('f', 6), ('g', 7))
    >>> tuple(zip('abcdefg', exrange(-1, '-inf', -1)))
    (('a', -1), ('b', -2), ('c', -3), ('d', -4), ('e', -5), ('f', -6), ('g', -7))
    '''

    start = 0
    step = 1

    if len(args) == 1:
        [stop] = args
    elif len(args) == 2:
        start, stop = args
    else:
        assert len(args) == 3
        start, stop, step = args

    if stop in [float('inf'), float('-inf'), 'inf', '-inf']:
        return itertools.count(start, step)
    else:
        return range(start, stop, step)


def get_elem(coll, indices):
    '''
    Example:
    >>> tensor = [[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]],
    ...           [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]]
    >>> get_elem(tensor, [0, 2])
    [8, 9, 10, 11]
    '''
    if isinstance(indices, int):
        indices = (indices,)
    elem = coll
    for idx in indices:
        elem = elem[idx]
    return elem


def set_elem(coll, indices, value):
    '''
    Example:
    >>> tensor = [[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]],
    ...           [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]]
    >>> set_elem(tensor, [0, 2], [0, 0, 0, 0])
    >>> tensor
    [[[0, 1, 2, 3], [4, 5, 6, 7], [0, 0, 0, 0]], [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]]
    '''
    if isinstance(indices, int):
        indices = (indices,)
    elem = coll
    for idx in indices[:-1]:
        elem = elem[idx]
    elem[indices[-1]] = value


def nest(first, *others):
    '''
    Example 1:

    >>> tuple(nest(range(2), 'abc'))
    ((0, 'a'), (0, 'b'), (0, 'c'), (1, 'a'), (1, 'b'), (1, 'c'))

    Example 2:

    >>> tensor = [[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]],
    ...           [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]]
    >>> size = [2, 3, 4]
    >>> tuple(get_elem(tensor, indices) for indices in nest(*map(range, size[:-1])))
    ([0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11])
    '''
    if others:
        for first_item in first:
            for other_items in nest(*others):
                yield (first_item,) + other_items
    else:
        for first_item in first:
            yield (first_item,)


def chunk_sizes(total_num, num_chunks):
    '''
    Example:

    >>> chunks = tuple(chunk_sizes(20, 6))
    >>> chunks
    (4, 4, 3, 3, 3, 3)
    >>> sum(chunks)
    20
    '''
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


def _max_idx_value_pairs(items, *, key=identity):
    assert len(items) > 0

    item_enum = enumerate(items)
    first_idx, first_item = first_idx_item = next(item_enum)

    max_idx_item_pairs = [first_idx_item]
    max_key = key(first_item)

    for idx_item_pair in item_enum:
        idx, item = idx_item_pair
        item_key = key(item)
        if item_key > max_key:
            max_idx_item_pairs = [idx_item_pair]
            max_key = item_key
        elif item_key < max_key:
            pass
        else:
            assert item_key == max_key
            max_idx_item_pairs.append(idx_item_pair)

    return max_idx_item_pairs


def maxall(items, *, key=identity):
    idx_item_pairs = _max_idx_value_pairs(items, key=key)
    indices, items = zip(*idx_item_pairs)
    return items


def minall(items, *, key=identity):
    return maxall(items, key=lambda x: -key(x))


def idxmaxall(items, *, key=identity):
    idx_item_pairs = _max_idx_value_pairs(items, key=key)
    indices, items = zip(*idx_item_pairs)
    return indices


def idxminall(items, *, key=identity):
    return idxmaxall(items, key=lambda x: -key(x))


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
    '''
    Example:

    >>> items = ['a', 'b', 'c', 'd', 'e', 'f']
    >>> items
    ['a', 'b', 'c', 'd', 'e', 'f']
    >>> replace_with_last(items, 1)
    'b'
    >>> items
    ['a', 'f', 'c', 'd', 'e']
    '''
    item = items[idx]
    items[idx] = items[-1]
    items.pop()
    return item


def indices_to_slices(indices, including_end_index=False, seq_length=None):
    '''
    Example:

    >>> tuple(indices_to_slices([5, 15], including_end_index=False))
    (slice(0, 5, None), slice(5, 15, None), slice(15, None, None))
    '''
    last_idx = 0
    for idx in indices:
        yield slice(last_idx, idx)
        last_idx = idx

    if including_end_index:
        if seq_length is not None:
            assert idx == seq_length
    else:
        yield slice(last_idx, None)


def split_by_indices(seq, indices, including_end_index=False):
    '''
    Example:

    >>> tuple(split_by_indices(tuple(range(20)), [5, 15], including_end_index=False))
    ((0, 1, 2, 3, 4), (5, 6, 7, 8, 9, 10, 11, 12, 13, 14), (15, 16, 17, 18, 19))
    '''
    return map(seq.__getitem__, indices_to_slices(indices, including_end_index=including_end_index, seq_length=len(seq)))


def lengths_to_slices(lengths):
    '''
    Example:

    >>> tuple(lengths_to_slices([5, 10, 5]))
    (slice(0, 5, None), slice(5, 15, None), slice(15, 20, None))
    '''
    indices = []
    accumulated_length = 0
    for length in lengths:
        accumulated_length += length
        indices.append(accumulated_length)

    return indices_to_slices(indices, including_end_index=True, seq_length=accumulated_length)


def split_by_lengths(seq, lengths):
    '''
    Example:

    >>> tuple(split_by_lengths(tuple(range(20)), [5, 10, 5]))
    ((0, 1, 2, 3, 4), (5, 6, 7, 8, 9, 10, 11, 12, 13, 14), (15, 16, 17, 18, 19))
    '''

    return map(seq.__getitem__, lengths_to_slices(lengths))


def partition(seq, n, fill_value=NO_VALUE, strict=True):
    # Similar to Hy's partition
    # https://github.com/hylang/hy/blob/0.18.0/hy/core/language.hy
    '''
    Example:

    >>> tuple(partition('abcdefgh', 2))
    (('a', 'b'), ('c', 'd'), ('e', 'f'), ('g', 'h'))
    >>> tuple(partition('abcdefg', 2, fill_value=None))
    (('a', 'b'), ('c', 'd'), ('e', 'f'), ('g', None))
    >>> tuple(partition('abcdefg', 2, strict=False))
    (('a', 'b'), ('c', 'd'), ('e', 'f'), ('g',))
    '''

    assert isinstance(n, int)
    assert n > 0

    if n == 1:
        for item in seq:
            yield (item,)
    else:
        iterator = iterate(seq)

        while iterator:
            first_item = next(iterator)
            items = [first_item]

            if fill_value is NO_VALUE:
                for _ in range(n - 1):
                    try:
                        items.append(next(iterator))
                    except StopIteration:
                        if strict:
                            raise Exception('The total number of items is not divided by {}'.format(n))
                        else:
                            break
            else:
                for _ in range(n - 1):
                    try:
                        items.append(next(iterator))
                    except StopIteration:
                        for _ in range(n - len(items)):
                            items.append(fill_value)
                        break
            yield tuple(items)


def flatten(coll, coll_type=(list, tuple, set)):
    # Similar to Hy's flatten
    # https://github.com/hylang/hy/blob/0.18.0/hy/core/language.hy
    '''
    Fallten objects in `coll` where the types of objects are in `coll_type`.

    Example:

    >>> flatten([0, [1, 2, [3, [4, 5], 6], 7, 8], 9])
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    '''

    assert not isinstance(coll, str), COLL_TYPE_STR_ERROR_MESSAGE
    assert is_type(coll_type)

    # if coll_type is None:
    #     coll_type = type(coll)
    # elif isinstance(coll_type, list):
    #     coll_type = tuple(coll_type)

    flattened = []

    def recurse(obj):
        if isinstance(obj, coll_type):
            for elem in obj:
                recurse(elem)
        else:
            flattened.append(obj)

    recurse(coll)
    return flattened


def firstelem(coll):
    # Similar to Hy's first
    # https://github.com/hylang/hy/blob/0.18.0/hy/core/language.hy
    '''
    Example:

    >>> firstelem(x for x in range(5))
    0
    '''
    return next(iter(coll))


def lastelem(coll):
    # Similar to Hy's last
    # https://github.com/hylang/hy/blob/0.18.0/hy/core/language.hy
    '''
    Example:

    >>> lastelem(x for x in range(5))
    4
    '''
    try:
        return next(reversed(coll))
    except TypeError:
        for elem in coll:
            pass
        return elem


def dropfirstk(coll, count):
    # Similar to Hy's drop
    # https://github.com/hylang/hy/blob/0.18.0/hy/core/language.hy

    '''
    Example:
    '''

    return itertools.islice(coll, count, None)


def chainelems(coll):
    '''
    Example:

    >>> tuple(chainelems([[1, 2, 3], 'abcdef', 'ABCDEF']))
    (1, 2, 3, 'a', 'b', 'c', 'd', 'e', 'f', 'A', 'B', 'C', 'D', 'E', 'F')
    '''
    for elem in coll:
        for item in elem:
            yield item


def repeat_in_order(coll, num_repeats):
    '''
    Example:

    >>> tuple(repeat_in_order([1, 2, 3, 4,], 3))
    (1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4)
    '''
    for elem in coll:
        for i in range(num_repeats):
            yield elem


def reversed_range(length):
    return range(length - 1, -1, -1)


def reversed_enumerate(coll, length=None):
    length = length or len(coll)
    return zip(reversed_range(length), reversed(coll))


if importlib.util.find_spec('tqdm') is not None:
    from tqdm import tqdm

    def xtqdm(*args, desc_fn=None, **kwargs):
        '''
        Extended tqdm

        Example:
        >>> import time                                                   # doctest: +SKIP
        >>> for idx in xtqdm(range(5), desc='no item is finished',        # doctest: +SKIP
        ...                  desc_fn=lambda: f'item {idx} is finished'):  # doctest: +SKIP
        ...     time.sleep(1)                                             # doctest: +SKIP
        '''

        pbar = tqdm(*args, **kwargs)

        if kwargs.get('desc') is None:
            if desc_fn is not None:
                pbar.set_description(desc_fn())

        for obj in pbar:
            yield obj
            if desc_fn is not None:
                pbar.set_description(desc_fn())

    def utqdm(*args, unit, update_fn, repr_format='{:}', init_repr=None, **kwargs):
        '''
        Unit tqdm

        Example:
        >>> import time                                       # doctest: +SKIP
        >>> numbers = list(range(10))                         # doctest: +SKIP
        >>> accum = 0                                         # doctest: +SKIP
        >>> for num in utqdm(numbers,                         # doctest: +SKIP
        ...                  unit='number',                   # doctest: +SKIP
        ...                  update_fn=lambda: accum,         # doctest: +SKIP
        ...                  repr_format='{:5.2f}',           # doctest: +SKIP
        ...                  init_repr='none'):               # doctest: +SKIP
        ...    time.sleep(1)                                  # doctest: +SKIP
        ...    accum += num                                   # doctest: +SKIP
        '''

        desc_kwargs = dict(not_none_valued_pairs(
            desc='{}: {}'.format(unit, init_repr) if init_repr is not None else None,
            desc_fn=lambda: ('{}: ' + repr_format).format(unit, update_fn())))

        return xtqdm(*args, **desc_kwargs, **kwargs)


def slice_by_max_size(items, size_fn, max_size):
    r"""
    Generate slices where the size of items in a slice is smallar than `max_size`.

    Example:
    >>> items = [10, 30, 50, 70, 90, 0, 20, 40, 60, 80]
    >>> slices = tuple(slice_by_max_size(items, lambda x: x, 100))
    >>> slices
    (slice(0, 3, None), slice(3, 4, None), slice(4, 6, None), slice(6, 8, None), slice(8, 9, None), slice(9, 10, None))
    >>> groups = tuple(items[slice] for slice in slices)
    >>> groups
    ([10, 30, 50], [70], [90, 0], [20, 40], [60], [80])
    """

    iterator = iterate(enumerate(map(size_fn, items)))

    last_index = 0
    group_size = 0

    if not iterator:
        # when items is empty
        yield from ()
    else:
        while iterator:
            index, size = next(iterator)
            assert size <= max_size, 'The size of an item exceeds the limit'

            if group_size + size <= max_size:
                group_size += size
            else:
                iterator.restore((index, size))
                yield slice(last_index, index)
                last_index = index
                group_size = 0

        assert group_size <= max_size
        yield slice(last_index, index + 1)
        last_index = index + 1
        group_size = 0


def split_by_max_size(items, size_fn, max_size):
    """
    Split items where the size of a split is smallar than `max_size`.

    Example:
    >>> items = [10, 30, 50, 70, 90, 0, 20, 40, 60, 80]
    >>> splits = tuple(split_by_max_size(items, lambda x: x, 100))
    >>> splits
    ([10, 30, 50], [70], [90, 0], [20, 40], [60], [80])
    """

    _data_source = items if hasattr(items, '__getitem__') else tuple(items)
    return map(
        _data_source.__getitem__,
        slice_by_max_size(
            items=_data_source,
            size_fn=size_fn,
            max_size=max_size
        ))


def iterfirstk(sequence, k):
    '''
    Example:

    >>> items = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
    >>> output = []
    >>> for x in iterfirstk(items, 3):
    ...     output.append(x)
    >>> print(output)
    ['a', 'b', 'c']
    '''
    iterator = iter(sequence)
    for idx in range(k):
        yield next(iterator)


