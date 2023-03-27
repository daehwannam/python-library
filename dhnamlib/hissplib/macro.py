
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


# the code for prelude is copied from "hissp.basic"
prelude_code = \
    ('from functools import partial,reduce\n'
     'from itertools import *;from operator import *\n'
     'def entuple(*xs):return xs\n'
     'def enlist(*xs):return[*xs]\n'
     'def enset(*xs):return{{*xs}}\n'
     "def enfrost(*xs):return __import__('builtins').frozenset(xs)\n"
     'def endict(*kvs):return{{k:i.__next__()for i in[kvs.__iter__()]for k in i}}\n'
     "def enstr(*xs):return''.join(''.__class__(x)for x in xs)\n"
     'def engarde(xs,h,f,/,*a,**kw):\n'
     ' try:return f(*a,**kw)\n'
     ' except xs as e:return h(e)\n'
     "_macro_=__import__('types').SimpleNamespace()\n"
     "try:exec('from {}._macro_ import *',vars(_macro_))\n"
     'except ModuleNotFoundError:pass').format(
         'hissp.basic')


def prelude():
    globals = inspect.stack()[1][0].f_globals
    exec(prelude_code, globals)


def load_macro(module, macro_names):
    globals = inspect.stack()[1][0].f_globals
    global_macro = globals['_macro_']
    for macro_name in macro_names:
        setattr(global_macro,
                macro_name,
                getattr(module._macro_, macro_name))


def load_all_macros(module):
    load_macro(module, dir(module._macro_))
