
import random


def add_assign_dict(d1, d2):
    for k, v in d2.items():
        d1[k] = d1.get(k, 0) + v


def div_dict(d, number):
    return type(d)([k, v / number] for k, v in d.items())


def shuffled(items, seed=None):
    if seed is not None:
        _random = random.Random(seed)
    else:
        _random = random
    new_items = list(items)
    _random.shuffle(new_items)
    return new_items
