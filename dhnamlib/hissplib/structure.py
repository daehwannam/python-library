
# from .decoration import parse_hy_args
from ..pylib.lisp import parse_lisp_args
from ..pylib.iteration import partition


def mapkv(*symbols):
    """
    >>> from dhnamlib.hissplib.compiler import eval_lissp
    >>> eval_lissp('''(mapkv :key-first "value-first" :key-second 'value-second)''')
    {'key-first': 'value-first', 'key-second': 'valueQzH_second'}
    >>> eval_lissp('''(mapkv 10 "a" 20 "b")''')
    {10: 'a', 20: 'b'}
    """
    args, kwargs = parse_lisp_args(symbols)
    pairs = tuple(partition(args, 2))
    try:
        return dict(pairs, **kwargs)
    except TypeError:
        breakpoint()
        return dict(pairs, **kwargs)

def tup(*args):
    return args
