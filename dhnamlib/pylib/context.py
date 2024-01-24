
import inspect
import functools
import contextlib
from contextlib import contextmanager  # https://realpython.com/python-with-statement/#creating-function-based-context-managers
import os

from .lazy import LazyEval, eval_lazy_obj, get_eval_obj_unless_lazy
from .decoration import deprecated
from .klass import Interface

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

    >>> env = Environment(a=100)
    >>>
    >>> # placeholders with default values
    >>> @env
    ... def func(u, w, x=env.ph.c(5), y=env.ph(5), z=env.ph(LazyEval(my_sum, 1, 2, 3, 4, 5))):
    ...     return u + w + x + y + z
    >>>
    >>> with env(b=3, y=10):
    ...     # `env.y` is set to 10, so the default value of parameter `y` in `func` is not used
    ...     # 100 + 3 + 5 + 10 + my_sum(1, 2, 3, 4, 5) == 33
    ...     print(func(u=env.a, w=env.b))
    ...     print(func(u=env.a, w=env.b))
    Calling my_sum
    133
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
        raise EnvironmentAttributeError(f'no such variable "{name}" in the local_env')

    def __setattr__(self, name, value):
        if self._setattr_enabled:
            super().__setattr__(name, value)
        else:
            # raise EnvironmentAttributeError("env variables can only be set by `with Environment.let()` or `Environment.update`")
            raise EnvironmentAttributeError("Attributes can only be set by `with Environment.let()`")

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

    def get(self, name, default=None):
        if name in self:
            return self.__getattr__(name)
        else:
            return default

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


class EnvironmentAttributeError(AttributeError):
    pass


_reserved_name_exceptoin_format_str = '"{name} is a reserved name'


class _EnvironmentBlock:
    def __init__(self, stack, pairs, kwargs):
        self._stack = stack
        self._dict = dict(pairs, **kwargs)

    def __enter__(self):
        self._stack.append(self._dict)

    def __exit__(self, except_type, except_value, except_traceback):
        self._stack.pop()


class _PlaceholderFactory:
    def __init__(self, env: Environment):
        self.env = env

    def __getattr__(self, name):
        return _Placeholder(self.env, name)

    def __call__(self, default_value):
        return _PlaceholderFactoryWithDefault(self.env, default_value)


class _PlaceholderFactoryWithDefault(_PlaceholderFactory):
    interface = Interface(_PlaceholderFactory)

    @interface.override
    def __init__(self, env: Environment, default_value):
        super().__init__(env)
        self.default_value = default_value

    @interface.override
    def __getattr__(self, name):
        return _PlaceholderWithDefaultValue(self.env, name, self.default_value)


class _Placeholder:
    def __init__(self, env, name):
        self.env = env
        self.name = name

    def get_value(self):
        return self.env.__getattr__(self.name)

    def __call__(self, default_value):
        return _PlaceholderWithDefaultValue(self.env, self.name, default_value)


class _PlaceholderWithDefaultValue(_Placeholder):
    interface = Interface(_Placeholder)

    @interface.override
    def __init__(self, env, name, default_value):
        super().__init__(env, name)
        self.default_value = default_value

    @interface.override
    def get_value(self):
        try:
            return super().get_value()
        except EnvironmentAttributeError:
            return eval_lazy_obj(self.default_value)



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

    def __exit__(self, except_type, except_value, except_traceback):
        pass


block = _Block()


class contextless:
    '''
    It does not perform any additional operation.

    Example:

    >>> with contextless():
    ...   numbers = list(range(5))
    ...   numbers = [x * 2 for x in numbers]
    ...
    >>> print(numbers)
    [0, 2, 4, 6, 8]
    '''

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        pass

    def __exit__(self, except_type, except_value, except_traceback):
        pass


class skippable:
    '''
    It does not perform any additional operation.

    Example:

    >>> saving = False
    >>> context = open if saving else skippable
    >>> file_path = 'test.txt'

    >>> with context(file_path, 'w') as f:
    ...   skip_if_possible(f)
    ...   print('Before saving text.')
    ...   f.write('Some text to be written.')
    '''

    def __init__(self, *args, forcing_to_skip=False, **kwargs):
        assert 'forcing_to_skip' not in kwargs
        self.forcing_to_skip = forcing_to_skip
        # pass

    def __enter__(self):
        return _SKIPPABLE_OBJ

    def __exit__(self, except_type, except_value, except_traceback):
        if self.forcing_to_skip:
            assert except_type is _SkippableError, "When forcing_to_skip == False, skip_if_possible should be called"

        if except_type is _SkippableError:
            return True
        else:
            return False


class _SkippableError(Exception):
    pass


_SKIPPABLE_OBJ = object()


def skip_if_possible(obj):
    if obj is _SKIPPABLE_OBJ:
        raise _SkippableError


class must_skipped(skippable):
    def __init__(self, *args, **kwargs):
        assert 'forcing_to_skip' not in kwargs
        super().__init__(*args, **kwargs, forcing_to_skip=True)


@contextmanager
def suppress_stdout():
    # https://stackoverflow.com/a/46129367
    with open(os.devnull, "w") as f:
        with contextlib.redirect_stdout(f):
            yield


@contextmanager
def suppress_stderr():
    # https://stackoverflow.com/a/46129367
    with open(os.devnull, "w") as f:
        with contextlib.redirect_stderr(f):
            yield


@contextmanager
def context_nest(*context_managers):
    """
    Example:

    Nest context managers

    >>> with context_nest(suppress_stdout(), suppress_stderr()) as (context_manager_1, context_manager_2):
    ...     print("This is not printed")
    """
    with contextlib.ExitStack() as exit_stack:
        for context_manager in context_managers:
            exit_stack.enter_context(context_manager)
        yield context_managers
