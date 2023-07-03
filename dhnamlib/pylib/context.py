
import inspect
import functools
import tempfile
import shutil
import os
# from contextlib import contextmanager  # https://realpython.com/python-with-statement/#creating-function-based-context-managers

from .lazy import LazyEval, eval_lazy_obj, get_eval_obj_unless_lazy
from .decoration import deprecated
from . import filesys

# Environment
# modified from https://stackoverflow.com/a/2002140/6710003

class Environment:
    """
    Example:

    >>> # Set the default values
    >>> env = Environment(a=100)
    >>>
    >>> # Placeholders as default arguments
    >>> # e.g. explicit naming -> env.ph.c / implicit naming -> env.ph
    >>> @env
    ... def func(u, w, x=env.ph.c, y=env.ph, z=env.ph):
    ...     return u + w + x + y + z
    >>>
    >>> def my_sum(*args):
    ...    print('Calling my_sum')
    ...    return sum(args)
    >>>
    >>> # Set values in a dynamic env
    >>> with env(b=3, c=5, y=10, z=LazyEval(my_sum, 1, 2, 3, 4, 5)):
    ...     print('Before the 1st function call')
    ...     print(func(u=env.a, w=env.b))
    ...     print('Before the 2nd function call')
    ...     print(func(u=env.a, w=env.b))
    ...     # Override value fo 'a' in the inner env
    ...     with env(a=1000):
    ...         print('Before the 3rd function call')
    ...         print(func(u=env.a, w=env.b))
    ...     print('Before the 4th function call')
    ...     print(func(u=env.a, w=env.b))
    Before the 1st function call
    Calling my_sum
    133
    Before the 2nd function call
    133
    Before the 3rd function call
    1033
    Before the 4th function call
    133
    """
    _setattr_enabled = True

    def __init__(self, pairs=(), **kwargs):
        self._stack = []
        _dict = dict(pairs, **kwargs)
        if _dict:
            # default values
            self._stack.append(_dict)
        self._reserved_names = ['ph']

        self.ph = _PlaceholderFactory(self)

        # self._setattr_enabled should be set at the end
        self._setattr_enabled = False

    def __getattr__(self, name):
        if name in self._reserved_names:
            raise Exception(_reserved_name_exceptoin_format_str.format(name=name))
        for local_env in reversed(self._stack):
            if name in local_env:
                value = local_env[name]
                return eval_lazy_obj(value)
        raise AttributeError(f'no such variable "{name}" in the local_env')

    def __setattr__(self, name, value):
        if self._setattr_enabled:
            super().__setattr__(name, value)
        else:
            # raise AttributeError("env variables can only be set by `with Environment.let()` or `Environment.update`")
            raise AttributeError("Attributes can only be set by `with Environment.let()`")

    def let(self, pairs=(), **kwargs):
        # context manager for a dynamic env
        for reserved_name in self._reserved_names:
            if reserved_name in kwargs:
                raise Exception(_reserved_name_exceptoin_format_str.format(name=reserved_name))

        return _EnvironmentBlock(self._stack, pairs, kwargs)

    @deprecated
    def update(self, pairs=(), **kwargs):
        self._stack[-1].update(pairs, **kwargs)

    def items(self, lazy=True):
        eval_obj_unless_lazy = get_eval_obj_unless_lazy(lazy)
        keys = set()
        for local_env in reversed(self._stack):
            for key, value in local_env.items():
                if key not in keys:
                    keys.add(key)
                    yield key, eval_obj_unless_lazy(value)

    def __contains__(self, name):
        for local_env in reversed(self._stack):
            if name in local_env:
                return True
        else:
            return False

    def decorate(self, func):
        signature = inspect.signature(func)

        def generate_ph_info_tuples():
            for idx, (name, param) in enumerate(signature.parameters.items()):
                if (
                        param.default is not inspect.Parameter.empty and
                        (isinstance(param.default, _Placeholder) or
                         isinstance(param.default, _PlaceholderFactory)) and
                        param.default.env == self
                ):
                    if param.kind == inspect._ParameterKind.POSITIONAL_OR_KEYWORD:
                        pos_idx = idx
                    else:
                        assert param.kind == inspect._ParameterKind.KEYWORD_ONLY
                        pos_idx = None
                    if isinstance(param.default, _Placeholder):
                        default = param.default
                    else:
                        assert isinstance(param.default, _PlaceholderFactory)
                        default = param.default.__getattr__(name)
                    yield pos_idx, name, default

        ph_info_tuples = tuple(generate_ph_info_tuples())

        if len(ph_info_tuples) == 0:
            raise Exception('no placeholder is used')

        @functools.wraps(func)
        def new_func(*args, **kwargs):
            new_kwargs = dict(kwargs)
            for pos_idx, name, placeholder in ph_info_tuples:
                if (pos_idx is None or len(args) <= pos_idx) and name not in kwargs:
                    new_kwargs[name] = placeholder.get_value()
            return func(*args, **new_kwargs)

        return new_func

    def __call__(self, *args, **kwargs):
        if len(args) > 0:
            # when used as a decorator
            assert len(args) == 1
            func = args[0]
            assert callable(func)
            assert len(kwargs) == 0
            return self.decorate(func)
        else:
            # when used as a context manager
            return self.let(**kwargs)


_reserved_name_exceptoin_format_str = '"{name} is a reserved name'


class _EnvironmentBlock:
    def __init__(self, stack, pairs, kwargs):
        self._stack = stack
        self._dict = dict(pairs, **kwargs)

    def __enter__(self):
        self._stack.append(self._dict)

    def __exit__(self, exc_type, exc_value, exc_tb):
        self._stack.pop()


class _PlaceholderFactory:
    def __init__(self, env: Environment):
        self.env = env

    def __getattr__(self, name):
        return _Placeholder(self.env, name)

class _Placeholder:
    def __init__(self, env, name):
        self.env = env
        self.name = name

    def get_value(self):
        return self.env.__getattr__(self.name)


# block
class _Block:
    '''
    It's used only to indent code for readability.
    It does not perform any additional operation.

    Example:

    >>> with block:
    ...   numbers = list(range(5))
    ...   numbers = [x * 2 for x in numbers]
    ...
    >>> print(numbers)
    [0, 2, 4, 6, 8]
    '''

    def __init__(self):
        pass

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, exc_tb):
        pass


block = _Block()


class _ReplaceDirectory:
    def __init__(self, dir_path, force=False):
        self.dir_path = dir_path
        self.force = force

    def __enter__(self):
        if not os.path.isdir(self.dir_path):
            if os.path.isfile(self.dir_path):
                raise Exception(f'"{self.dir_path}" is a file rather than a directory')
            elif self.force:
                os.makedirs(self.dir_path)
            else:
                raise Exception(f'"{self.dir_path}" does not exist')
        parent_dir_path = filesys.get_parent_path(self.dir_path)
        self.temp_dir_path = tempfile.mkdtemp(dir=parent_dir_path)
        dir_octal_mode = filesys.get_octal_mode(self.temp_dir_path)
        filesys.set_octal_mode(self.temp_dir_path, dir_octal_mode)
        return self.temp_dir_path

    def __exit__(self, exc_type, exc_value, exc_tb):
        shutil.rmtree(self.dir_path)
        os.rename(self.temp_dir_path, self.dir_path)


def replace_dir(dir_path, force=False):
    '''
    :param dir_path: The path to a directory
    :param force: If force=True, ignore existence of the directory. Otherwise raise exception.

    Example:

    >>> dir_path = 'some-dir'
    >>> os.makedirs(dir_path)
    >>> with open(os.path.join(dir_path, 'some-file-1'), 'w') as f:
    ...     pass
    ...
    >>> os.listdir(dir_path)
    ['some-file-1']
    >>> with replace_dir(dir_path) as temp_dir_path:
    ...     with open(os.path.join(temp_dir_path, 'some-file-2'), 'w') as f:
    ...         pass
    ...
    >>> os.listdir(dir_path)
    ['some-file-2']
    >>> shutil.rmtree(dir_path)  # remove the directory
    '''
    return _ReplaceDirectory(dir_path, force=force)


def copy_dir(source, target, replacing=False, overwriting=False):
    if replacing:
        shutil.rmtree(target)
    return shutil.copytree(source, target, dirs_exist_ok=overwriting)
