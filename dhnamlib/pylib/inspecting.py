
import inspect
from itertools import chain


def _get_scope(level=0):
    '''
    >>> def foo(a=1, b=2, level=0):
    ...     c = 3
    ...     def bar(d=4):
    ...         e = 5
    ...         f = 6
    ...         def baz(g=7):
    ...             h=8
    ...             return _get_scope(level)
    ...         return baz()
    ...     return bar()

    >>> scope = foo()
    >>> scope['a']
    1
    >>> scope['d']
    4

    '''

    # This does not work with lambda functions.

    step = 1
    stack = inspect.stack()
    last_frame = stack[step + level][0]
    f_locals_list = [last_frame.f_locals]

    step += 1
    while step + level < len(stack):
        frame = stack[step + level][0]
        if last_frame.f_code.co_name in frame.f_locals:
            f_locals_list.append(frame.f_locals)
            last_frame = frame
        else:
            break
        step += 1

    local_scope = dict(chain.from_iterable(f_locals.items() for f_locals in reversed(f_locals_list)))
    return local_scope
