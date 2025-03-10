
import inspect
from importlib.metadata import version

from hissp.munger import munge
from packaging.version import Version


# Note
# hissp.basic._macro_ is used in hissp=0.3.0
# hissp.macros._macro_ is used in hissp=0.5.0

if Version(version('hissp')) >= Version('0.4.0'):
    MACRO_PACKAGE = 'hissp.macros'
else:
    MACRO_PACKAGE = 'hissp.basic'

all_macro_import_code = \
f"""
try:
    from {MACRO_PACKAGE} import _macro_ as _basic_macro_
    if '_macro_' not in locals():
        _macro_ = __import__('types').SimpleNamespace()
    _macro_ = __import__('types').SimpleNamespace(**vars(_macro_), **vars(_basic_macro_))
except ModuleNotFoundError:
    pass
"""


def import_all_basic_macros():
    """
    Update the global variable _macro_ to include basic macros.
    """

    globals = inspect.stack()[1][0].f_globals
    exec(all_macro_import_code, globals)


# the code for prelude is copied from "hissp.basic" of "hissp=0.3.0"
# prelude_code = \
#     ('from functools import partial,reduce\n'
#      'from itertools import *;from operator import *\n'
#      'def entuple(*xs):return xs\n'
#      'def enlist(*xs):return[*xs]\n'
#      'def enset(*xs):return{{*xs}}\n'
#      "def enfrost(*xs):return __import__('builtins').frozenset(xs)\n"
#      'def endict(*kvs):return{{k:i.__next__()for i in[kvs.__iter__()]for k in i}}\n'
#      "def enstr(*xs):return''.join(''.__class__(x)for x in xs)\n"
#      'def engarde(xs,h,f,/,*a,**kw):\n'
#      ' try:return f(*a,**kw)\n'
#      ' except xs as e:return h(e)\n'
#      "_macro_=__import__('types').SimpleNamespace()\n"
#      "try:exec('from {}._macro_ import *',vars(_macro_))\n"
#      'except ModuleNotFoundError:pass').format(
#          MACRO_PACKAGE)

prelude_code = \
    ('from functools import partial, reduce\n'
     'from itertools import *\n'
     'from operator import *\n')


def prelude():
    """
    Import functions and macros globally.
    """

    globals = inspect.stack()[1][0].f_globals
    exec(prelude_code, globals)
    exec(all_macro_import_code, globals)


def load_macro(module, macro_name, alias=None):
    """
    >>> from dhnamlib.hissplib.module import import_lissp
    >>> from dhnamlib.hissplib.macro import load_macro
    >>> from dhnamlib.hissplib.compiler import eval_lissp

    >>> lissplib_base = import_lissp('dhnamlib.hissplib.lissplib.base')
    >>> load_macro(lissplib_base, 'el-let', 'let')

    >>> eval_lissp('(let ((x 10) (y 20)) (print x y))')
    10 20
    """
    if alias is None:
        alias = macro_name

    globals = inspect.stack()[1][0].f_globals
    if '_macro_' not in globals:
        globals['_macro_'] = __import__('types').SimpleNamespace()
    global_macro = globals['_macro_']
    munged_macro_name = munge(macro_name)
    munged_alias = munge(alias)
    setattr(global_macro,
            munged_alias,
            getattr(module._macro_, munged_macro_name))


def load_macros(module, macro_names):
    globals = inspect.stack()[1][0].f_globals
    global_macro = globals['_macro_']
    for macro_name in macro_names:
        munged_macro_name = munge(macro_name)
        setattr(global_macro,
                munged_macro_name,
                getattr(module._macro_, munged_macro_name))


def load_all_macros(module):
    load_macros(module, dir(module._macro_))
