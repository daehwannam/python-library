
import inspect
import functools
# from contextlib import contextmanager  # https://realpython.com/python-with-statement/#creating-function-based-context-managers

# Scope
# modified from https://stackoverflow.com/a/2002140/6710003

class Scope:
    """
    Example 1:

    >>> scope = Scope()
    >>>
    >>> with scope(a=10, b=20):
    ...     with scope(a=20, c= 30):
    ...         print(scope.a, scope.b, scope.c)
    ...     print(scope.a, scope.b)
    20 20 30
    10 20

    Example 2:

    >>> scope = Scope()
    >>>
    >>> @scope
    >>> def func(x=scope.ph.a, y=scope.ph.b):
    ...     return x + y
    >>>
    >>> with scope(a=10, b=20):
    ...     print(func())
    30

    Example 3:

    >>> scope = Scope()
    >>>
    >>> @scope
    >>> def func(x=scope.ph, y=scope.ph):
    ...     return x + y
    >>>
    >>> with scope(x=10, y=20):
    ...     print(func())
    30
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
        for scope in reversed(self._stack):
            if name in scope:
                return scope[name]
        raise AttributeError("no such variable in the scope")

    def __setattr__(self, name, value):
        if self._setattr_enabled:
            super().__setattr__(name, value)
        else:
            raise AttributeError("scope variables can only be set using `with Scope.let()`")

    def let(self, **kwargs):
        for reserved_name in self._reserved_names:
            if reserved_name in kwargs:
                raise Exception(f'"{reserved_name}" is a reserved name')

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
            assert len(args) == 1
            func = args[0]
            assert callable(func)
            assert len(kwargs) == 0
            return self.decorate(func)
        else:
            return self.let(**kwargs)


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
