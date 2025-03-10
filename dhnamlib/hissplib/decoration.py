
import functools

from hissp.munger import munge

from ..pylib.lisp import parse_lisp_args


def parse_hy_args(symbols):
    hy_args, hy_kwargs = parse_lisp_args(symbols)
    new_hy_kwargs = dict([k.replace('-', '_'), v] for k, v in hy_kwargs.items())
    return hy_args, new_hy_kwargs


def hy_function(func):
    '''
    >>> from dhnamlib.hissplib.compiler import eval_lissp

    >>> @hy_function
    ... def func(a_b_c, d_e_f):
    ...     return a_b_c + d_e_f

    >>> eval_lissp('(func 100 :d-e-f 50)')
    150
    >>> eval_lissp('(func 100 :d_e_f 50)')
    150
    '''    
    @functools.wraps(func)
    def new_func(*args):
        # assert len(kwargs) == 0
        new_args, new_kwargs = parse_hy_args(args)
        return func(*new_args, **new_kwargs)

    return new_func
