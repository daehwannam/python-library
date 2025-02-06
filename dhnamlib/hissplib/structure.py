
from .decoration import parse_hy_args


def mapkv(*symbols):
    """
    >>> from dhnamlib.hissplib.compile import eval_lissp
    >>> eval_lissp('''(mapkv :key-first "value-first" :key-second 'value-second)''')
    {'key_first': 'value-first', 'key_second': 'valueQzH_second'}
    """
    args, kwargs = parse_hy_args(symbols)
    return dict(*args, **kwargs)

def tup(*args):
    return args
