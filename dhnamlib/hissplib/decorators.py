
import functools
from ..pylib.lisp import parse_hy_args as _parse_hy_args


def parse_hy_args(symbols):
    hy_args, hy_kwargs = _parse_hy_args(symbols)
    new_hy_kwargs = dict([k.replace('-', '_'), v] for k, v in hy_kwargs.items())
    return hy_args, new_hy_kwargs


def hy_function(func):
    @functools.wraps(func)
    def new_func(*args, **kwargs):
        assert len(kwargs) == 0
        new_args, new_kwargs = parse_hy_args(args)
        return func(*new_args, **new_kwargs)

    return new_func
