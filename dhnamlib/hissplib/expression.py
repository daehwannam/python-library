
import json

from hissp.munger import demunge
from hissp.compiler import MAYBE

MAIN = '__main__'
backquoted_symbol_prefixes = [
    f'{MAIN}{MAYBE}',
    f'{MAIN}..',
    'builtins..',
]

def remove_backquoted_symbol_prefixes(expr):
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
    Example:

    >>> repr_as_hash_str('some "text" here')
    '#"some \\"text\\" here"'

    Without '#' character, a string becomes a "raw string" in Lissp (Hissp)
    #"text..." is called a "hash string" in Lissp.

    Hash string is useful especially to express a string that contains special characters,
    such as double quotes, or unicode characters.

    In addition, a hash string representation should be encoded by `json.dumps` to be evaluated.

    >>> import json
    >>> from dhnamlib.hissplib.compile import eval_lissp
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

    >>> from dhnamlib.hissplib.compile import lissp_to_hissp
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
