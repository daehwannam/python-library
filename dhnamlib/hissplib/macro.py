
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


def load_macro(module, *macro_names):
    globals = inspect.stack()[1][0].f_globals
    global_macro = globals['_macro_']
    for macro_name in macro_names:
        setattr(global_macro,
                macro_name,
                getattr(module._macro_, macro_name))


def load_all_macros(module):
    load_macro(module, dir(module))
