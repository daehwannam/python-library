
import sys
import resource
import os
import time
import signal
from contextlib import contextmanager
import linecache

from . import decoration


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
    exc_type, exc_obj, tb = sys.exc_info()
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
