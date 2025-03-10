
import re
import inspect

from hissp.munger import munge

# from ..pylib.lisp import parse_hy_args
from .expression import munge_lisp_args

kw_regex = re.compile(r'{([^\s{}]+)}')


def symbolic_format(template, *args, kw_regex=kw_regex, base_level=0):
    '''
    >>> from dhnamlib.hissplib.compiler import eval_lissp
    >>> aQzH_bQzH_c = 'ABC'  # munge("'a-b-c") == 'aQzH_bQzH_c'
    >>> eval_lissp('(symbolic_format "{1} {0} {1} {a-b-c} {x-y-z} {i-j-k}" "One" "Zero" :x-y-z "XYZ" :i-j-k None)')
    'Zero One Zero ABC XYZ {i-j-k}'

    >>> from dhnamlib.hissplib.macro import load_macro
    >>> from dhnamlib.hissplib.module import import_lissp

    >>> lissplib_base = import_lissp('dhnamlib.hissplib.lissplib.base')
    >>> load_macro(lissplib_base, 'el-let', 'let')

    >>> eval_lissp('(let ((a-b-c "A~B~C")) (symbolic_format "{1} {0} {1} {a-b-c} {x-y-z} {i-j-k}" "One" "Zero" :x-y-z "XYZ" :i-j-k None))')
    'Zero One Zero A~B~C XYZ {i-j-k}'

    # >>> eval_lissp('(let ((a-b-c "A~B~C")) (let ((x-y-z "X~Y~Z")) (symbolic_format "{1} {0} {1} {a-b-c} {x-y-z} {i-j-k}" "One" "Zero" :i-j-k None)))')
    # 'Zero One Zero ABC X~Y~Z {i-j-k}'
    '''
    assert base_level >= 0

    hy_args, hy_kwargs = munge_lisp_args(args)

    splits = []
    index = 0
    arg_index = 0

    # f_globals = inspect.stack()[2 + base_level][0].f_globals
    # f_locals = inspect.stack()[2 + base_level][0].f_locals

    # def get_value(kw):
    #     try:
    #         return eval(kw, f_globals, f_locals)
    #     except NameError:
    #         return None

    def get_from_f_globals(kw):
        return inspect.stack()[2 + base_level][0].f_globals.get(kw)

    def get_from_f_locals(kw):
        return inspect.stack()[2 + base_level][0].f_locals.get(kw)

    for match_obj in kw_regex.finditer(template):
        span = match_obj.span()
        splits.append(template[index: span[0]])
        kw = match_obj.group(1)
        if kw == '':
            replaced = hy_args[arg_index]
            arg_index += 1
        else:
            if arg_index > 0:
                raise ValueError('cannot switch from automatic field numbering to manual field specification.')

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
                    # replaced = get_value(var_kw)

            assert replaced is not None, f'The keyword "{kw}" cannot be instantiated in the template "{template}."'
            assert isinstance(replaced, str), f'The replacement of the keyword "{kw}" is an object of the type {type(replaced)} rather than the type str.'

        splits.append(replaced)
        index = span[1]

    splits.append(template[index:])

    return ''.join(splits)
    # try:
    #     return ''.join(splits)
    # except TypeError:
    #     breakpoint()
    #     print(len(splits))


# regex for string type = re.compile(r'"(?:\\.|[^"\\])*"')
# https://stackoverflow.com/a/16130746

_preprocess_format_expr_regex = re.compile(r'\(format\s+("(?:\\.|[^"\\])*")\)')

def preprocess_format_expr(text):
    """
    >>> from dhnamlib.hissplib.compiler import eval_lissp
    >>> from dhnamlib.hissplib.macro import prelude
    >>> from dhnamlib.hissplib.operation import import_operators

    >>> prelude()
    >>> import_operators()

    >>> first_name = 'John'
    >>> last_name = 'Smith'
    >>> country = 'USA'

    >>> text = r'''(+ (symbolic_format "My name is \"{first_name} {last_name}\"") (symbolic_format ", and I'm from \"{country}\"."))'''

    >>> preprocess_format_expr(text)  # doctest: +SKIP
    '(+ (symbolic_format "My name is \\"{first_name} {last_name}\\"") (symbolic_format ", and I\'m from \\"{country}\\"."))'  # doctest: +SKIP
    >>> eval_lissp(preprocess_format_expr(text))  # doctest: +SKIP
    'My name is "John Smith", and I\'m from "USA".'  # doctest: +SKIP

    """
    splits = []
    index = 0

    for template_match_obj in _preprocess_format_expr_regex.finditer(text):
        group_text = template_match_obj.group(1)

        full_span = template_match_obj.span()
        group_span = template_match_obj.span(1)

        splits.append(text[index: group_span[1]])

        keywords = tuple(kw_match_obj.group(1) for kw_match_obj in kw_regex.finditer(group_text))
        keywords_expr = ' ' + ' '.join(f':{keyword} {keyword}' for keyword in keywords)
        splits.append(keywords_expr)
        splits.append(text[group_span[1]: full_span[1]])

        index = full_span[1]

    splits.append(text[index:])

    preprocessed_text = ''.join(splits)

    return preprocessed_text
