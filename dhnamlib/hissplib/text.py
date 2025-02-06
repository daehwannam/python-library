
import re
import inspect

from .decoration import parse_hy_args

kw_regex = re.compile(r'{([^\s{}]+)}')


def hy_format(string, *args, kw_regex=kw_regex):
    '''
    >>> from dhnamlib.hissplib.compile import eval_lissp
    >>> a_b_c = 'ABC'
    >>> eval_lissp('(hy_format "{1} {0} {1} {a-b-c} {x-y-z} {i-j-k}" "One" "Two" :x-y-z "XYZ" :i-j-k None)')
    'Two One Two ABC XYZ {i_j_k}'
    '''
    hy_args, hy_kwargs = parse_hy_args(args)

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
                var_kw = kw.replace('-', '_')
                if var_kw in hy_kwargs:
                    var_value = hy_kwargs[var_kw]
                    if var_value is None:
                        # When a value of a keyword is None,
                        # the placeholder remains as it is.
                        replaced = f'{{{var_kw}}}'
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

