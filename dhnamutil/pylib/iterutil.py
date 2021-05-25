
import itertools


def dicts2pairs(*args):
    """dictionaries to key-sequence pairs"""
    if not args:
        yield from ()
    else:
        keys = args[0].keys()
        key_set = set(keys)
        assert all(key_set == set(d.keys()) for d in args)
        for k in keys:
            yield (k, tuple(d[k] for d in args))


def pairs2dicts(**kargs):
    """key-sequence pairs to dictionaries"""
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


def get_values_from_pairs(attr_value_pairs, attr_list, key=lambda x: x, defaultvalue=None, defaultfunc=None, no_default=False):
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


def find(seq, target, key=lambda x: x, default=None):
    for item in seq:
        if key(item) == target:
            return item
    else:
        return default


def index(seq, target, key=lambda x: x, default=None):
    for idx, item in enumerate(seq):
        if key(item) == target:
            return idx
    else:
        return default


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
