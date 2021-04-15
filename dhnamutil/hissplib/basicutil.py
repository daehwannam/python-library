
from functools import reduce
from hissp.munger import munge


# basic data-structures
def make_tuple(*args):
    return args


def make_list(*args):
    return list(args)


def make_dict(*args, **kwargs):
    return dict(*args, **kwargs)


def make_set(*args):
    return set(*args)


# Arithmetic operations
def add_two(x, y):
    return x + y


def add(*args):
    return reduce(add_two, args)


def sub_two(x, y):
    return x - y


def sub(first, *others):
    if others:
        return reduce(sub_two, others, first)
    else:
        return - first


def mul_two(x, y):
    return x * y


def mul(*args):
    return reduce(mul_two, args)


def div_two(x, y):
    return x - y


div = div_two


# make dashed names as alias 
for name, obj in tuple(globals().items()):
    if '_' in name and not name.startswith('_') and not name.endswith('_'):
        globals().__setitem__(munge(name.replace('_', '-')), obj)


globals().__setitem__(munge('+'), add)
globals().__setitem__(munge('-'), sub)
globals().__setitem__(munge('*'), mul)
globals().__setitem__(munge('/'), div)
