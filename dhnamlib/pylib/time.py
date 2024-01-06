
import time
import signal
import datetime
from contextlib import contextmanager


class TimeoutException(Exception):
    pass


@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


def time_limit_example():
    try:
        with time_limit(3):
            time.sleep(2)
    except TimeoutException as e:
        print("Timed out!")


def time_limit_example2():
    with time_limit(2):
        time.sleep(3)


class TimeMeasure:
    '''
    Example:

    >>> tm = TimeMeasure()
    >>> tm.check()
    >>> type(tm.elapse())
    <class 'float'>

    >>> with TimeMeasure() as tm:
    ...   time.sleep(0.5)
    ...
    >>> abs(tm.interval - 0.5) < 0.1
    True

    >>> tm = TimeMeasure()
    >>> with tm:
    ...   time.sleep(0.5)
    ...
    >>> abs(tm.interval - 0.5) < 0.1
    True
    '''

    def check(self):
        "Update a checkpoint."
        self.checkpoint = time.time()

    def elapse(self):
        "Elapse time (seconds) from the last checkpoint."
        return time.time() - self.checkpoint

    def __enter__(self):
        self.check()
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self._interval = self.elapse()

    @property
    def interval(self):
        return self._interval


def get_time_seed(max_num_digits=6):
    exponential_num = 10 ** max_num_digits
    curr_time = time.time_ns()
    time_seed = curr_time - (curr_time // exponential_num) * exponential_num
    return time_seed


def old_get_ymdhms(dt: datetime.datetime = None):
    def td(num):
        '''Convert a number to a two-digit string'''
        # https://stackoverflow.com/a/15509693
        return '{:02d}'.format(num)

    if dt is None:
        dt = datetime.datetime.now()

    return '{year}{month}{day}.{hour}{minute}{second}'.format(
        year=dt.year, month=td(dt.month), day=td(dt.day),
        hour=td(dt.hour), minute=td(dt.minute), second=td(dt.second))


def get_YmdHMSf(dt: datetime.datetime = None):
    if dt is None:
        dt = datetime.datetime.now()

    initial_date_str = dt.strftime("%Y-%m-%d_%H:%M:%S_%f")
    return initial_date_str
