
import time
import signal
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
    def check(self):
        self.checkpoint = time.time()

    def elapsed(self):
        return time.time() - self.checkpoint
