
import re
import inspect

from hissp.munger import munge

# from ..pylib.lisp import parse_hy_args
from .expression import munge_lisp_args

kw_regex = re.compile(r'{([^\s{}]+)}')


def symbolic_format(string, *args, kw_regex=kw_regex):
    '''
    >>> from dhnamlib.hissplib.compile import eval_lissp
    >>> aQzH_bQzH_c = 'ABC'  # munge("'a-b-c") == 'aQzH_bQzH_cQzAPOS_'
    >>> eval_lissp('(symbolic_format "{1} {0} {1} {a-b-c} {x-y-z} {i-j-k}" "One" "Zero" :x-y-z "XYZ" :i-j-k None)')
    'Zero One Zero ABC XYZ {i-j-k}'

    >>> from dhnamlib.hissplib.macro import load_macro
    >>> from dhnamlib.hissplib.module import import_lissp

    >>> hissplib_basic = import_lissp('dhnamlib.hissplib.basic')
    >>> load_macro(hissplib_basic, 'el-let', 'let')

    >>> eval_lissp('(let ((a-b-c "A~B~C")) (symbolic_format "{1} {0} {1} {a-b-c} {x-y-z} {i-j-k}" "One" "Zero" :x-y-z "XYZ" :i-j-k None))')
    'Zero One Zero A~B~C XYZ {i-j-k}'
    '''
    hy_args, hy_kwargs = munge_lisp_args(args)

    splits = []
    index = 0
    arg_index = 0

    def get_from_f_globals(kw):
        return inspect.stack()[2][0].f_globals.get(kw)

    def get_from_f_locals(kw):
        return inspect.stack()[2][0].f_locals.get(kw)

    for match_obj in kw_regex.finditer(string):
        span = match_obj.span()
        splits.append(string[index: span[0]])
        kw = match_obj.group(1)
        if kw == '':
            replaced = hy_args[arg_index]
            arg_index += 1
        else:
            if arg_index > 0:
                raise ValueError('cannot switch from automatic field numbering to manual field specification')

            if kw.isnumeric():
                replaced = hy_args[int(kw)]
            else:
                # var_kw = kw.replace('-', '_')
                var_kw = munge(kw)
                if var_kw in hy_kwargs:
                    var_value = hy_kwargs[var_kw]
                    if var_value is None:
                        # When a value of a keyword is None,
                        # the placeholder remains as it is.
                        replaced = f'{{{kw}}}'
                    else:
                        replaced = var_value
                else:
                    replaced = (get_from_f_locals(var_kw) or get_from_f_globals(var_kw))

            if replaced is None:
                print(splits)
            assert replaced is not None

        splits.append(replaced)
        index = span[1]

    splits.append(string[index:])

    return ''.join(splits)

