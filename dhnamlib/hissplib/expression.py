
from hissp.compiler import MAYBE

MAIN = '__main__'
backquoted_symbol_prefixes = [
    f'{MAIN}{MAYBE}',
    f'{MAIN}..',
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
