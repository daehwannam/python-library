
import sys
import doctest
import importlib
import argparse

def run_test():
    '''
    Example:

    $ python -m dhnamlib.pylib.test -v dhnamlib.pylib.iteration
    '''

    # This code is modified from doctest._test

    parser = argparse.ArgumentParser(description="doctest runner")
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='print very verbose output for all tests')
    parser.add_argument('-o', '--option', action='append',
                        choices=doctest.OPTIONFLAGS_BY_NAME.keys(), default=[],
                        help=('specify a doctest option flag to apply'
                              ' to the test run; may be specified more'
                              ' than once to apply multiple options'))
    parser.add_argument('-f', '--fail-fast', action='store_true',
                        help=('stop running tests after first failure (this'
                              ' is a shorthand for -o FAIL_FAST, and is'
                              ' in addition to any other -o options)'))
    parser.add_argument('module', nargs='+',
                        help='module to test')
    args = parser.parse_args()
    testmodules = args.module
    # Verbose used to be handled by the "inspect argv" magic in DocTestRunner,
    # but since we are using argparse we are passing it manually now.
    verbose = args.verbose
    options = 0
    for option in args.option:
        options |= doctest.OPTIONFLAGS_BY_NAME[option]
    if args.fail_fast:
        options |= doctest.FAIL_FAST
    for modulename in testmodules:
        m = importlib.import_module(modulename)
        print(modulename)
        print(m)
        failures, _ = doctest.testmod(m, verbose=verbose, optionflags=options)
        if failures:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(run_test())
