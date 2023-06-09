
import inspect
import functools
# from contextlib import contextmanager  # https://realpython.com/python-with-statement/#creating-function-based-context-managers

from .lazy import LazyEval

# Scope
# modified from https://stackoverflow.com/a/2002140/6710003

class Scope:
    """
    Example:

    >>> # Set the default values
    >>> scope = Scope(a=100)
    >>>
    >>> # Placeholders as default arguments
    >>> # e.g. explicit naming -> scope.ph.c / implicit naming -> scope.ph
    >>> @scope
    >>> def func(u, w, x=scope.ph.c, y=scope.ph, z=scope.ph):
    ...     return u + w + x + y + z
    >>>
    >>> def my_sum(*args):
    ...    print('Calling my_sum')
    ...    return sum(args)
    >>>
    >>> # Set values in a dynamic scope
    >>> with scope(b=3, c=5, y=10, z=LazyEval(my_sum, 1, 2, 3, 4, 5)):
    ...     print('Before the 1st function call')
    ...     print(func(u=scope.a, w=scope.b))
    ...     print('Before the 2nd function call')
    ...     print(func(u=scope.a, w=scope.b))
    ...     # Override value fo 'a' in the inner scope
    ...     with scope(a=1000):
    ...         print('Before the 3rd function call')
    ...         print(func(u=scope.a, w=scope.b))
    ...     print('Before the 4th function call')
    ...     print(func(u=scope.a, w=scope.b))
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

    def __init__(self, **kwargs):
        self._stack = []
        if kwargs:
            # default values
            self._stack.append(kwargs)
        self._reserved_names = ['ph']

        self.ph = _PlaceholderFactory(self)

        # self._setattr_enabled should be set at the end
        self._setattr_enabled = False

    def __getattr__(self, name):
        if name in self._reserved_names:
            raise Exception(_reserved_name_exceptoin_format_str.format(name=name))
        for scope in reversed(self._stack):
            if name in scope:
                value = scope[name]
                if isinstance(value, LazyEval):
                    return value.get()
                else:
                    return value
        raise AttributeError(f'no such variable "{name}" in the scope')

    def __setattr__(self, name, value):
        if self._setattr_enabled:
            super().__setattr__(name, value)
        else:
            raise AttributeError("scope variables can only be set using `with Scope.let()`")

    def let(self, **kwargs):
        # context manager for a dynamic scope
        for reserved_name in self._reserved_names:
            if reserved_name in kwargs:
                raise Exception(_reserved_name_exceptoin_format_str.format(name=reserved_name))

        return _ScopeBlock(self._stack, kwargs)

    def decorate(self, func):
        signature = inspect.signature(func)

        def generate_ph_info_tuples():
            for idx, (name, param) in enumerate(signature.parameters.items()):
                if (
                        param.default is not inspect.Parameter.empty and
                        (isinstance(param.default, _Placeholder) or
                         isinstance(param.default, _PlaceholderFactory)) and
                        param.default.scope == self
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


class _ScopeBlock:
    def __init__(self, stack, kwargs):
        self._stack = stack
        self.kwargs = kwargs

    def __enter__(self):
        self._stack.append(self.kwargs)

    def __exit__(self, exc_type, exc_value, exc_tb):
        self._stack.pop()


class _PlaceholderFactory:
    def __init__(self, scope: Scope):
        self.scope = scope

    def __getattr__(self, name):
        return _Placeholder(self.scope, name)

class _Placeholder:
    def __init__(self, scope, name):
        self.scope = scope
        self.name = name

    def get_value(self):
        return self.scope.__getattr__(self.name)


# block
class _Block:
    def __init__(self):
        pass

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, exc_tb):
        pass


block = _Block()
