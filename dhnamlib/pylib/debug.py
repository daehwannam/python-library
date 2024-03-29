
import sys
import resource
import os
import time
import signal
from contextlib import contextmanager
import linecache
import inspect

from . import decoration
from . import iteration
from .text import split_into_vars


# import psutil


# def get_ram_usage_rate():
#     # # https://stackoverflow.com/a/42275253
#     # tot_m, used_m, free_m = map(int, os.popen('free -t -m').readlines()[-1].split()[1:])
#     # return used_m / tot_m

#     # https://stackoverflow.com/a/38984517
#     return psutil.virtual_memory()[2] / 100


@contextmanager
def periodic_checking(period, predicate, post_process=lambda: None, exception=Exception):
    def signal_handler(signum, frame):
        if predicate():
            signal.alarm(period)
            post_process()
        else:
            raise exception
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(period)

    try:
        yield
    finally:
        signal.alarm(0)


def periodic_laugh(seconds=3):
    laugh_limit = [3]  # because int type is immutable

    def predicate():
        if laugh_limit[0] > 0:
            laugh_limit[0] -= 1
            return True
        else:
            return False

    with periodic_checking(seconds, predicate, lambda: print('hahaha')):
        try:
            while True:
                time.sleep(1)
                print('woohoo')
        except Exception:
            print('{} times of "hahaha"'.format(laugh_limit[0]))


class MemoryLimitException(Exception):
    pass


# def periodic_memory_check_example(max_ram_usage_rate=0.4, seconds=3):
#     def predicate():
#         return get_ram_usage_rate() < max_ram_usage_rate

#     with periodic_checking(seconds, predicate,
#                            lambda: print('memory isn\'t overused yet'),
#                            MemoryLimitException):
#         memory = [None] * 1024
#         try:
#             for _ in range(1000):
#                 memory *= 2
#                 time.sleep(1)
#                 print('doing...')
#         except MemoryLimitException:
#             print('MemoryLimitException occurs')
#         else:
#             print('No exception')



# python memory limit utility
# https://stackoverflow.com/a/41125461
def memory_limit(proportion):
    assert 0 < proportion < 1
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (get_memory() * 1024 * proportion, hard))


def get_memory():
    with open('/proc/meminfo', 'r') as mem:
        free_memory = 0
        for i in mem:
            sline = i.split()
            if str(sline[0]) in ('MemFree:', 'Buffers:', 'Cached:'):
                free_memory += int(sline[1])
    return free_memory


def memory_limist_test():
    memory_limit(0.5)  # Limitates maximun memory usage to half
    try:
        import time
        data = list(range(8))
        while True:
            time.sleep(1)
            data.extend(data)
            print('extended')
    except MemoryError:
        sys.stderr.write('\n\nERROR: Memory Exception\n')
        sys.exit(1)


def get_exception_string():
    except_type, exc_obj, tb = sys.exc_info()
    f = tb.tb_frame
    lineno = tb.tb_lineno
    filename = f.f_code.co_filename
    linecache.checkcache(filename)
    line = linecache.getline(filename, lineno, f.f_globals)
    return 'Exception occurred in (path {}, line {}, code "{}"): {}'.format(
        filename, lineno, line.strip(), repr(exc_obj))


def print_exception():
    return get_exception_string()


@decoration.cache
def print_warning(msg):
    print("Warning: {}".format(msg))


class VariableTracker:
    '''
    >>> var_tracker = VariableTracker()

    >>> a = 10
    >>> b = 20
    >>> var_tracker.register(a, b)

    >>> print(var_tracker.has_any_existing_var())
    True
    >>> print(sorted(var_tracker.get_existing_var_names()))
    ['a', 'b']

    >>> del a
    >>> print(var_tracker.has_any_existing_var())
    True
    >>> print(sorted(var_tracker.get_existing_var_names()))
    ['b']

    >>> del b
    >>> print(var_tracker.has_any_existing_var())
    False
    >>> print(sorted(var_tracker.get_existing_var_names()))
    []

    >>> var_tracker.assert_no_existing_variable()
    '''

    def __init__(self):
        self.tracked_var_names = set()
        self._frame_depth = 2

    def register(self, *objs):
        var_names = vars2names(*objs, frame_stack=inspect.stack())
        self.register_var_names(var_names, scope=self.get_default_scope())

    def unregister(self, *objs):
        var_names = vars2names(*objs, frame_stack=inspect.stack())
        self.unregister_var_names(var_names, scope=self.get_default_scope())

    def register_var_names(self, var_names, scope=None):
        if len(var_names) == 1 and (' ' in var_names[0] or ',' in var_names[0]):
            # _var_names = var_names
            var_names = split_into_vars(var_names[0])

        if scope is None:
            scope = self.get_default_scope()

        for var_name in var_names:
            assert var_name not in self.tracked_var_names, f'The variable {var_name} is already registered'
            assert var_name in scope, f'The variable {var_name} was not defined'
            self.tracked_var_names.add(var_name)

    def unregister_var_names(self, var_names):
        if len(var_names) == 1 and (' ' in var_names[0] or ',' in var_names[0]):
            # _var_names = var_names
            var_names = split_into_vars(var_names[0])

        for var_name in var_names:
            assert var_name in self.tracked_var_names, f'The variable {var_name} was not registered'
            self.tracked_var_names.remove(var_name)

    @decoration.deprecated
    def register_obj(self, obj, scope=None):
        if scope is None:
            scope = self.get_default_scope()

        for var_name, var_obj in scope.items():
            if var_obj is obj:
                break
        else:
            raise Exception('The object is not assigned to a varaible')

        self.tracked_var_names.add(var_name)

    @decoration.deprecated
    def unregister_obj(self, obj, scope=None):
        if scope is None:
            scope = self.get_default_scope()

        for var_name, var_obj in scope.items():
            if var_obj is obj:
                break
        else:
            raise Exception('The object is not assigned to a varaible')

        self.tracked_var_names.remove(var_name)

    def assert_no_existing_variable(self):
        scope = self.get_default_scope()

        if self.has_any_existing_var(scope=scope):
            raise Exception('There is an existing variable: {}'.format(', '.join(self.get_existing_var_names(scope=scope))))

    def has_any_existing_var(self, scope=None):
        if scope is None:
            scope = self.get_default_scope()

        for var_name in self.tracked_var_names:
            if var_name in scope:
                return True
        else:
            return False

    def get_existing_var_names(self, scope=None):
        if scope is None:
            scope = self.get_default_scope()

        var_names = []

        for var_name in self.tracked_var_names:
            if var_name in scope:
                var_names.append(var_name)

        return var_names

    def get_default_scope(self, offset=0):
        scope = inspect.stack()[self._frame_depth + offset][0].f_locals
        return scope


def vars2names(*variables, frame_stack=None):
    '''
    Convert variables to their names.
    The code that calls this function should be one line.

    >>> some_var = 100
    >>> vars2names(some_var)
    ['some_var']
    >>> another_var = 100
    >>> vars2names(some_var, another_var)
    ['some_var', 'another_var']
    '''

    # Reference:
    # https://stackoverflow.com/a/53684586

    # assert len(variables) > 0

    try:
        if frame_stack is None:
            frame_stack = inspect.stack()

        code = frame_stack[1].code_context[0]
        # Caution: `code_context` has only one line
        fn_name = frame_stack[0].frame.f_code.co_name  # it is 'vars2names' or another function name
        # fn_name = frame_stack[0].function  # it is 'vars2names' or another function name

        fn_start_idx = code.index(fn_name)
        content_start_idx = code.index("(", fn_start_idx + len(fn_name)) + 1
        content_end_idx = code.index(")", content_start_idx)

        var_names = split_into_vars(code[content_start_idx: content_end_idx].strip())
        # if len(var_names) == 1:
        #     return var_names[0]
        # else:
        #     return var_names

        return var_names

    except ValueError:
        raise Exception('Parsing code failed')


def NIE(msg=None):
    """Not Implementated Error"""

    args = (msg,) if msg is not None else ()
    raise NotImplementedError(*args)
