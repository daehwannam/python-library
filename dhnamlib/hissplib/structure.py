
# from .decoration import parse_hy_args
from ..pylib.lisp import parse_lisp_args


def mapkv(*symbols):
    """
    >>> from dhnamlib.hissplib.compile import eval_lissp
    >>> eval_lissp('''(mapkv :key-first "value-first" :key-second 'value-second)''')
    {'key-first': 'value-first', 'key-second': 'valueQzH_second'}
    """
    args, kwargs = parse_lisp_args(symbols)
    return dict(*args, **kwargs)

def tup(*args):
    return args
