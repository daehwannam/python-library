
import inspect


all_macro_import_code = \
"""
try:
    from hissp.basic import _macro_
    _macro_ = __import__('types').SimpleNamespace(**vars(_macro_))
except ModuleNotFoundError:
    pass
"""


def import_all_basic_macros():
    globals = inspect.stack()[1][0].f_globals
    exec(all_macro_import_code, globals)
