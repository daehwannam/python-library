
import functools
import inspect
import itertools

from . import filesys


# def cache(func):
#     """keep a cache of previous function calls.
#     it's similar to functools.lru_cache"""

#     cache_memory = {}

#     @functools.wraps(func)
#     def cached_func(*args, **kwargs):
#         cache_key = args + tuple(kwargs.items())
#         if cache_key not in cache_memory:
#             cache_memory[cache_key] = func(*args, **kwargs)
#         return cache_memory[cache_key]

#     cached_func.cache = cache_memory

#     return cached_func


def cache(func):
    return functools.lru_cache(maxsize=None)(func)


def cached_property(func):
    return property(cache(func))


# def singleton(func):
#     @functools.wraps(func)
#     def decorated_func(*args, **kwargs):
#         if decorated_func.singleton is None:
#             decorated_func.singleton = func(*args, **kwargs)
#         else:
#             raise Exception('This function is already evaluated.')
#         return decorated_func.singleton

#     decorated_func.singleton = None

#     return decorated_func


def unnecessary(func):
    called = False

    @functools.wraps(func)
    def new_func(*args, **kwargs):
        nonlocal called
        if not called:
            called = True
            print("Unneccessary function {} is called.".format(func.__name__))
        return func(*args, **kwargs)

    return new_func


def abstractfunction(func):
    @functools.wraps(func)
    def new_func(*args, **kwargs):
        raise NotImplementedError(
            "The abstract function '{}' is not implemented".format(func.__name__))
    return new_func



# def load_after_save(file_path_arg_name, *, load_func, save_func):
#     import os

#     def load_after_save_decorator(func):
#         @functools.wraps(func)
#         def load_after_save_func(*args, **kwargs):
#             file_path = kwargs[file_path_arg_name]
#             del kwargs[file_path_arg_name]
#             if not os.path.isfile(file_path):
#                 save_func(file_path, func(*args, **kwargs))
#             return load_func(file_path)

#         return load_after_save_func

#     return load_after_save_decorator


save_load_pair_dict = dict(
    pickle=(filesys.pickle_save, filesys.pickle_load),
    json=(filesys.json_save, filesys.json_load),
    json_pretty=(filesys.json_pretty_save, filesys.json_load),
    extended_json=(filesys.extended_json_save, filesys.extended_json_load),
    extended_json_pretty=(filesys.extended_json_pretty_save, filesys.extended_json_load),
)


def file_cache(file_path_arg_name='file_cache_path', *, save_fn=None, load_fn=None, format='extended_json_pretty'):
    '''
    Example 1
    >>> from dhnamlib.pylib.filesys import json_save, json_load
    >>> from dhnamlib.pylib.decorators import file_cache
    >>>
    >>> @file_cache('cache_path', save_fn=json_save, load_fn=json_load)
    >>> def make_dict_and_print(*pairs):
    >>>     d = dict(pairs)
    >>>     print('make_dict_and_print is called')
    >>>     return d
    >>>
    >>> pairs = [['a', 2], ['b', 4], ['c', 6]]
    >>> d1 = make_dict_and_print(*pairs, cache_path='./path/to/cache.json')
    >>> d2 = make_dict_and_print(*pairs, cache_path='./path/to/cache.json')
    >>> print(d1 == d2)

    Example 2
    >>> @file_cache(format='json')
    >>> def make_dict_and_print(*pairs):
    >>>     d = dict(pairs)
    >>>     print('make_dict_and_print is called')
    >>>     return d
    '''
    import os

    if format is not None:
        assert save_fn is load_fn is None
        save_fn, load_fn = save_load_pair_dict[format]

    def file_cache_decorator(func):
        @functools.wraps(func)
        def file_cache_func(*args, **kwargs):
            file_path = kwargs.get(file_path_arg_name)
            assert file_path is not None, f'\'{file_path_arg_name}\' is not designated'
            del kwargs[file_path_arg_name]

            if os.path.isfile(file_path):
                obj = load_fn(file_path)
            else:
                obj = func(*args, **kwargs)
                filesys.mkdirs_unless_exist(file_path)
                save_fn(obj, file_path)
            return obj

        return file_cache_func

    return file_cache_decorator


fcache = file_cache(format='extended_json_pretty')


def curry(func):
    '''
    Example
    >>> @curry
    >>> def func(a, b, c, *, d, e, f=6, g=7):
    >>>     return (a, b, c, d, e, f, g)
    >>>
    >>> print(func(1, 2)(3, 4, 5))
    >>> print(func(1, 2, 3, 4, 5))
    >>> print(func(1, 2, d=4)(3, e=5))
    >>> print(func(1, 2, d=4)(3, e=5, f=66))
    '''

    signature = inspect.signature(func)
    position_to_param_key = []
    positional_param_keys = set()
    reading_keyword_params = False
    for name, param in signature.parameters.items():
        param_str = str(param)
        if "=" in param_str or reading_keyword_params:
            # e.g. var='default-value'
            pass
        elif "*" == param_str:
            reading_keyword_params = True
        else:
            # e.g. *args and **kwargs are not allowed
            assert not param_str.startswith('*')
            positional_param_keys.add(name)
        position_to_param_key.append(name)

    param_dict = dict()

    def make_curried(_param_dict, positional_arg_count):
        def curried(*args, **kwargs):
            param_dict = dict(_param_dict)
            for idx, arg in enumerate(args, positional_arg_count):
                assert position_to_param_key[idx] not in param_dict
                param_dict[position_to_param_key[idx]] = arg

            for k, v in kwargs.items():
                assert k not in param_dict
                param_dict[k] = v

            if all(param_key in param_dict for param_key in positional_param_keys):
                return func(**param_dict)
            else:
                return make_curried(param_dict, positional_arg_count + len(args))
        return curried

    return make_curried(param_dict, 0)


# Register
class Register:
    '''
    Example
    >>> register = Register(strategy='lazy')
    >>> name_fn = register.retrieve('name-fn')
    >>>
    >>> @register('name-fn')
    >>> def full_name(first, last):
    >>>     return ' '.join([first, last])
    >>>
    >>> print(name_fn('John', 'Smith'))
    '''

    strategies = ['instant', 'lazy', 'conditional']

    def __init__(self, strategy='instant'):
        assert strategy in self.strategies
        self.strategy = strategy
        self.memory = dict()

    @staticmethod
    def _normalize_identifier(identifier):
        if isinstance(identifier, list):
            identifier = tuple(identifier)
        else:
            return identifier

    @curry
    def __call__(self, identifier, obj):
        identifier = self._normalize_identifier(identifier)

        assert identifier not in self.memory
        self.memory[identifier] = obj
        return obj

    # def __call__(self, identifier, func=None):
    #     if func is None:
    #         def decorator(func):
    #             assert identifier not in self.memory
    #             self.memory[identifier] = func
    #             return func
    #         return decorator
    #     else:
    #         assert identifier not in self.memory
    #         self.memory[identifier] = func

    def retrieve(self, identifier, strategy=None):
        identifier = self._normalize_identifier(identifier)

        if strategy is None:
            strategy = self.strategy
        else:
            assert strategy in self.strategies

        if (strategy == 'lazy') or (strategy == 'conditional' and identifier not in self.memory):
            registered = LazyValue(self, identifier)
        else:
            assert strategy in ['instant', 'conditional']
            registered = self.memory[identifier]

        return registered

    @staticmethod
    def _msg_not_registered(identifier):
        return f'"{identifier}" is not registered.'


class LazyValue:
    def __init__(self, register, identifier):
        self.register = register
        self.identifier = identifier

    def get(self):
        if self.identifier in self.register.memory:
            return self.register.memory[self.identifier]
        else:
            raise Exception(Register._msg_not_registered(self.identifier))

    def __call__(self, *args, **kwargs):
        if self.identifier in self.register.memory:
            return self.register.memory[self.identifier](*args, **kwargs)
        else:
            raise Exception(Register._msg_not_registered(self.identifier))


# Scope
# modified from https://stackoverflow.com/a/2002140/6710003

class Scope:
    """
    Example 1
    >>> scope = Scope()
    >>>
    >>> with scope(a=10, b=20):
    >>>     with scope(a=20, c= 30):
    >>>         print(scope.a, scope.b, scope.c)
    >>>     print(scope.a, scope.b)

    Example 2
    >>> scope = Scope()
    >>>
    >>> @scope
    >>> def func(x=scope.ph.a, y=scope.ph.b):
    >>>     return x + y
    >>>
    >>> with scope(a=10, b=20):
    >>>     print(func())

    """
    _setattr_enabled = True

    def __init__(self, stack=[]):
        self._stack = stack
        self._reserved_names = ['ph']

        self.ph = _PlaceholderFactory(self)

        # self._setattr_enabled should be set at the end
        self._setattr_enabled = False

    def __getattr__(self, name):
        for scope in reversed(self._stack):
            if name in scope:
                return scope[name]
        raise AttributeError("no such variable in environment")

    def __setattr__(self, name, value):
        if self._setattr_enabled:
            super().__setattr__(name, value)
        else:
            raise AttributeError("scope variables can only be set using `with Scope.let()`")

    def let(self, **kwargs):
        for reserved_name in self._reserved_names:
            if reserved_name in kwargs:
                raise Exception(f'"{reserved_name}" is a reserved name')

        return _EnvBlock(self._stack, kwargs)

    def decorate(self, func):
        signature = inspect.signature(func)

        def generate_ph_info_tuples():
            for idx, (name, param) in enumerate(signature.parameters.items()):
                if param.default is not inspect.Parameter.empty and \
                   isinstance(param.default, _Placeholder) and \
                   param.default.scope == self:
                    if param.kind == inspect._ParameterKind.POSITIONAL_OR_KEYWORD:
                        pos_idx = idx
                    else:
                        pos_idx = None
                    yield pos_idx, name, param.default

        ph_info_tuples = tuple(generate_ph_info_tuples())

        @functools.wraps(func)
        def new_func(*args, **kwargs):
            new_kwargs = dict(kwargs)
            for pos_idx, name, placeholder in ph_info_tuples:
                if (pos_idx is None or len(args) <= pos_idx) and name not in kwargs:
                    new_kwargs[name] = placeholder.get()
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


class _EnvBlock:
    def __init__(self, stack, kwargs):
        self._stack = stack
        self.kwargs = kwargs

    def __enter__(self):
        self._stack.append(self.kwargs)

    def __exit__(self, t, v, tb):
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

    def get(self):
        return self.scope.__getattr__(self.name)
