
import inspect
import operator
from functools import reduce
from hissp.munger import munge


# # basic data-structures
# "preluce" already includes entuple, enlist, ...

# def make_tuple(*args):
#     return args


# def make_list(*args):
#     return list(args)


# def make_dict(*args, **kwargs):
#     return dict(*args, **kwargs)


# def make_set(*args):
#     return set(*args)


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
    return x / y


div = div_two


# make dashed names as alias 
for name, obj in tuple(globals().items()):
    # without tuple wrapping globals(), globals() is changed during loop, then RuntimeError occurs.
    if '_' in name and not name.startswith('_') and not name.endswith('_'):
        globals().__setitem__(munge(name.replace('_', '-')), obj)

# allowing operator symbols
op_name_func_pairs = [['+', add],
                      ['-', sub],
                      ['*', mul],
                      ['/', div],
                      ['=', operator.eq],
                      ['!=', operator.ne],
                      ['<', operator.lt],
                      ['<=', operator.le],
                      ['>', operator.gt],
                      ['>=', operator.ge]]

for op_name, func in op_name_func_pairs:
    globals().__setitem__(munge(op_name), func)


def import_operators():
    """
    >>> from dhnamlib.hissplib.compiler import eval_lissp
    >>> import_operators()
    >>> eval_lissp('(+ "Hello" " " "World" "!")')
    'Hello World!'
    >>> eval_lissp('(- 100 10 5)')
    85
    """

    for op_name, func in op_name_func_pairs:
        inspect.stack()[1][0].f_locals.__setitem__(munge(op_name), func)
