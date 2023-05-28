
import functools
import inspect
import itertools
import warnings

from . import filesys


def curry(func, *args, **kwargs):
    '''
    Example
    >>> @curry
    >>> def func(a, b, c, *, d, e, f=6, g=7):
    >>>     return (a, b, c, d, e, f, g)
    >>>
    >>> print(func(1, 2, 3)(d=4, e=5))
    >>> print(func(1, 2)(3)(d=4, e=5))
    >>> print(func(1, 2, d=4)(3, e=5))
    >>> print(func(1, 2, d=4)(3, e=5, f=66))
    '''

    # [Note]
    # Implementation with functools.partial may be easier
    # e.g.
    # >>> from functools import partial
    # >>> f = lambda *args, **kwargs: (args, kwargs)
    # >>> partial(partial(partial(f, 1,2, a=10), 3,4, b=20), 5,6, c=30)()
    # ((1, 2, 3, 4, 5, 6), {'a': 10, 'b': 20, 'c': 30})

    signature = inspect.signature(func)
    position_to_param_key = []
    non_default_param_keys = set()
    reading_positional_params = True
    num_positional_only_params = 0
    for name, param in signature.parameters.items():
        if param.default is not inspect._empty:
            reading_positional_params = False
        else:
            non_default_param_keys.add(name)
            if param.kind is inspect._ParameterKind.KEYWORD_ONLY:
                reading_positional_params = False
            else:
                assert reading_positional_params
                # *args, **kwargs are disallowed
                # assert param.kind not in {inspect._ParameterKind.VAR_POSITIONAL, inspect._ParameterKind.VAR_KEYWORD}
                if param.kind is inspect._ParameterKind.POSITIONAL_ONLY:
                    num_positional_only_params += 1
                else:
                    param.kind is inspect._ParameterKind.POSITIONAL_OR_KEYWORD
                position_to_param_key.append(name)

    def make_curried(prev_args, prev_kwargs):
        def curried(*args, **kwargs):
            new_args = list(prev_args)
            for idx, arg in enumerate(args, len(prev_args)):
                assert idx < len(position_to_param_key)
                assert position_to_param_key[idx] not in prev_kwargs
                new_args.append(arg)

            new_kwargs = dict(prev_kwargs)
            for k, v in kwargs.items():
                assert k not in new_kwargs
                new_kwargs[k] = v

            def is_complete():
                if num_positional_only_params <= len(new_args):
                    arg_keys = set(itertools.chain(position_to_param_key[:len(new_args)], new_kwargs))
                    return arg_keys.issuperset(non_default_param_keys)
                else:
                    return False

            if is_complete():
                return func(*new_args, **new_kwargs)
            else:
                return make_curried(new_args, new_kwargs)
        return curried

    return make_curried(args, kwargs)


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


@curry
def cache(func, maxsize=None):
    return functools.lru_cache(maxsize)(func)


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


def deprecated(func):
    @functools.wraps(func)
    def new_func(*args, **kwargs):
        # https://docs.python.org/3/library/warnings.html
        warnings.warn('Deprecated function {} is called.'.format(func.__name__),
                      DeprecationWarning)
        return func(*args, **kwargs)

    return new_func


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

    STRATEGIES = ('instant', 'lazy', 'conditional')

    def __init__(self, strategy='instant'):
        assert strategy in self.STRATEGIES
        self.strategy = strategy
        self.memory = dict()

    @staticmethod
    def _normalize_identifier(identifier):
        if isinstance(identifier, list):
            identifier = tuple(identifier)
        return identifier

    @curry
    def __call__(self, identifier, obj):
        identifier = self._normalize_identifier(identifier)

        assert identifier not in self.memory, f'"{identifier}" is already registered.'
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
            assert strategy in self.STRATEGIES

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


@curry
def construct(construct_fn, func, /):
    '''
    Example

    >>> @construct(dict)
    ... def make_dict(key_seq, value_fn):
    ...     for key in key_seq:
    ...         yield key, value_fn(key)

    >>> make_dict(range(5), str)
    {0: '0', 1: '1', 2: '2', 3: '3', 4: '4'}
    '''

    @functools.wraps(func)
    def decorated_func(*args, **kwargs):
        return construct_fn(func(*args, **kwargs))

    return decorated_func


def variable(func):
    '''
    >>> @variable
    ... def num_list():
    ...    return [1, 2, 3, 4]

    >>> num_list
    [1, 2, 3, 4]
    '''
    return func()


def attribute(obj):
    '''
    >>> class A:
    ...     def __init__(self):
    ...         @attribute(self)
    ...         def num_list():
    ...             return [1, 2, 3, 4]

    >>> a = A()
    >>> a.num_list
    [1, 2, 3, 4]
    '''

    def decorator(func):
        setattr(obj, func.__name__, func())
        ns = inspect.stack()[1][0].f_locals
        if func.__name__ in ns:
            return ns[func.__name__]
        else:
            return None

    return decorator
