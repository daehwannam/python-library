
import inspect

from hissp.compiler import readerless
from hissp.reader import Lissp


def eval_lissp(code, ns=None):
    if ns is None:
        ns = inspect.stack()[1][0].f_globals
        # merging globals and locals could be a better choice
        # >>> inspect.stack()[1][0].f_locals

    lissp_parser = Lissp()  # A lissp object should not be shared by multiple threads
    hissp_form = next(lissp_parser.reads(code))
    py_code = readerless(hissp_form, ns=ns)
    return eval(py_code, ns)


if __name__ == '__main__':
    # TEST
    def remove_space(text):
        return ''.join(text.split())

    def add(*args):
        accum = args[0]
        for arg in args[1:]:
            accum += arg
        return accum

    def main():
        func = eval_lissp('(lambda (x) (add (remove_space x) (remove_space "a    b c") (remove_space "a b    c d   e")))')
        print(func('x      y z'))

    main()
