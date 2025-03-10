
import json

from hissp.munger import demunge, munge
from hissp.compiler import MAYBE

from ..pylib.lisp import parse_lisp_args


MAIN = '__main__'
# MAYBE = "..QzMaybe_."
backquoted_symbol_prefixes = [
    f'{MAIN}{MAYBE}',
    f'{MAIN}..',
    'builtins..',
]

def remove_backquoted_symbol_prefixes(expr):
    '''
    Example:

    >>> from dhnamlib.hissplib.compiler import eval_lissp

    >>> original = eval_lissp("`(+ + print list ,list 1 2 3)")
    >>> print(original)
    ('__main__..QzMaybe_.QzPLUS_', '__main__..QzPLUS_', 'builtins..print', 'builtins..list', <class 'list'>, 1, 2, 3)

    >>> removed = remove_backquoted_symbol_prefixes(original)
    >>> print(removed)
    ('QzPLUS_', 'QzPLUS_', 'print', 'list', <class 'list'>, 1, 2, 3)
    '''

    if isinstance(expr, str):
        for removable_symbol_prefix in backquoted_symbol_prefixes:
            if expr.startswith(removable_symbol_prefix):
                return expr[len(removable_symbol_prefix):]
        else:
            return expr
    elif isinstance(expr, tuple):
        return tuple(map(remove_backquoted_symbol_prefixes, expr))
    else:
        return expr


# def repr_as_raw_str(expr: str):
#     '''
#     Example:

#     >>> repr_as_raw_str('some text here')
#     '"some text here"'
#     '''

#     return json.dumps(expr)

def repr_as_hash_str(expr: str):
    r'''
    Note:
    Hash string was used in "hissp==0.3.0".
    However, it's not used in "hissp==0.5.0".
    In "hissp==0.5.0", normal string can represent special characters with backslash (e.g "\n").
    
    Example:

    >>> repr_as_hash_str('some "text" here')
    '#"some \\"text\\" here"'

    Without '#' character, a string becomes a "raw string" in Lissp (Hissp)
    #"text..." is called a "hash string" in Lissp.

    Hash string is useful especially to express a string that contains special characters,
    such as double quotes, or unicode characters.

    In addition, a hash string representation should be encoded by `json.dumps` to be evaluated.

    >>> import json
    >>> from dhnamlib.hissplib.compiler import eval_lissp
    >>> eval_lissp('(.upper #{})'.format(json.dumps('"π" is called PI.')))
    '"Π" IS CALLED PI.'

    That resemble Python's eval
    >>> eval("\"\\n\" + \"\\n\"")
    '\n\n'
    >>> eval('{} + {}'.format(json.dumps("\n"), json.dumps("\n")))
    '\n\n'
    '''

    return '#' + json.dumps(expr)


def demunge_recursively(expr):
    '''
    Example:

    >>> from dhnamlib.hissplib.compiler import lissp_to_hissp
    >>> expr = lissp_to_hissp("(outter-symbol-1 (inner-symbol-1 inner-symbol-2) outter-symbol-2)")
    >>> expr                    # doctest: +SKIP
    ('outterQz_symbolQz_1', ('innerQz_symbolQz_1', 'innerQz_symbolQz_2'), 'outterQz_symbolQz_2')  # doctest: +SKIP
    >>> demunge_recursively(expr)
    ('outter-symbol-1', ('inner-symbol-1', 'inner-symbol-2'), 'outter-symbol-2')
    '''

    if isinstance(expr, str):
        return demunge(expr)
    elif isinstance(expr, (list, tuple)):
        return tuple(map(demunge_recursively, expr))
    else:
        # e.g. int
        return expr


def munge_lisp_args(symbols):
    hy_args, hy_kwargs = parse_lisp_args(symbols)
    new_hy_kwargs = dict([munge(k), v] for k, v in hy_kwargs.items())
    return hy_args, new_hy_kwargs
