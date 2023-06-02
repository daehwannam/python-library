import cProfile
import inspect


def run_context(statement, globals=None, locals=None, filename=None, sort=-1):
    # Related:
    # - https://stackoverflow.com/a/8682791
    # - https://stackoverflow.com/a/4492582
    if globals is None and locals is None:
        globals = inspect.stack()[1][0].f_globals
        locals = inspect.stack()[1][0].f_locals

    return cProfile.runctx(
        statement=statement,
        globals=globals, locals=locals,
        filename=filename, sort=sort)
