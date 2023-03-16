
import inspect

from hissp.compiler import readerless
from hissp.reader import Lissp


def lissp_to_hissp(lissp_expr):
    lissp_parser = Lissp()  # A Lissp object should not be shared by multiple threads
    return next(lissp_parser.reads(lissp_expr))


def eval_lissp(code, ns=None):
    if ns is None:
        # merging globals and locals
        ns = {**inspect.stack()[1][0].f_globals,
              **inspect.stack()[1][0].f_locals}

    hissp_form = lissp_to_hissp(code)
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
