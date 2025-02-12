
from code import InteractiveInterpreter, InteractiveConsole


def interact_until_error(interpreter_cls=InteractiveInterpreter, banner=None, readfunc=None, local=None, exitmsg=None):
    """Closely emulate the interactive Python interpreter.

    This is a backwards compatible interface to the InteractiveInterpreter
    class.  When readfunc is not specified, it attempts to import the
    readline module to enable GNU readline if it is available.

    Arguments (all optional, all default to None):

    interpreter_cls -- a subclass of InteractiveInterpreter
    banner -- passed to interpreter_cls.interact()
    readfunc -- if not None, replaces interpreter_cls.raw_input()
    local -- passed to InteractiveInterpreter.__init__()
    exitmsg -- passed to interpreter_cls.interact()

    """
    # This function is moidified from 'code.interact'

    console = interpreter_cls(local)
    if readfunc is not None:
        console.raw_input = readfunc
    else:
        try:
            import readline
        except ImportError:
            pass
    console.interact(banner, exitmsg)


class ScriptInteractiveConsole(InteractiveConsole):
    def runcode(self, code):
        """Execute a code object.

        When an exception occurs, self.showtraceback() is called to
        display a traceback, then the program is ended.

        A note about KeyboardInterrupt: this exception may occur
        elsewhere in this code, and may not always be caught.  The
        caller should be prepared to deal with it.

        """
        try:
            exec(code, self.locals)
        except Exception:
            self.showtraceback()
            raise SystemExit


def interact_with_script(file_path, banner=None, local=None, exitmsg=None, adding_empty_first_line=False):
    with open(file_path) as f:
        empty_first_line_needed = True

        def readfunc(prompt):
            nonlocal empty_first_line_needed
            if adding_empty_first_line and empty_first_line_needed:
                empty_first_line_needed = False
                line = ''
            else:
                line = f.readline()  # All lines except the last end with "\n".

                if not line:
                    raise EOFError
                elif not line.endswith('\n'):
                    # Note:
                    # If the last line of the file is not empty,
                    # The line line can be code that does not end with "\n".
                    line += '\n'

            if line.strip():
                print(prompt, end='')
                print(line, end='')
            else:
                print()

            return line

        interact_until_error(
            interpreter_cls=ScriptInteractiveConsole,
            banner=banner if banner is not None else f'Started executing "{file_path}."\n',
            readfunc=readfunc,
            local=local,
            exitmsg=exitmsg if exitmsg is not None else f'Finished executing "{file_path}".\n'
        )


if __name__ == "__main__":
    """
    Example:
    >>> python -m dhnamlib.pylib.script_interacting path/to/code.py  # doctest: +SKIP
    """
    import argparse

    parser = argparse.ArgumentParser(description="script executor")
    parser.add_argument('file_path', help='path to file to execute')
    args = parser.parse_args()

    interact_with_script(args.file_path, banner='', exitmsg='', adding_empty_first_line=True)
