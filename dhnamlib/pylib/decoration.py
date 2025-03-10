
import functools
import inspect
import itertools
import warnings
from contextlib import ContextDecorator

from . import filesys
from .constant import NO_VALUE


def curry(func, *args, **kwargs):
    '''
    Example
    >>> @curry
    ... def func(a, b, c, *, d, e, f=6, g=7):
    ...     return (a, b, c, d, e, f, g)
    >>>
    >>> print(func(1, 2, 3)(d=4, e=5))
    (1, 2, 3, 4, 5, 6, 7)
    >>> print(func(1, 2)(3)(d=4, e=5))
    (1, 2, 3, 4, 5, 6, 7)
    >>> print(func(1, 2, d=4)(3, e=5))
    (1, 2, 3, 4, 5, 6, 7)
    >>> print(func(1, 2, d=4)(3, e=5, f=66))
    (1, 2, 3, 4, 5, 66, 7)
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
        @functools.wraps(func)
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


# def _cache(func):
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


def id_cache(func):
    """
    Keep a cache of previous function calls.
    It uses IDs of arguments for computing keys.
    """

    def key(*args, **kwargs):
        return tuple(itertools.chain(map(id, args), sorted((k, id(v)) for k, v in kwargs.items())))

    return keyed_cache(key, func)


@curry
def keyed_cache(key, func):
    """
    Cache with a custom key function.

    >>> @keyed_cache(lambda coll, target: (id(coll), target))
    ... def find(coll, target):
    ...     for idx, elem in enumerate(coll):
    ...         if elem == target:
    ...             return idx
    ...     else:
    ...         return None

    >>> large_tuple = tuple(range(1, 101))
    >>> find(large_tuple, 90)
    89
    >>> find(large_tuple, 90)
    89

    In the above example, the key function uses `id(coll)` to compute a key.
    Since `coll` can be a large object, using its id is efficient for computing the key.
    """

    cache_memory = {}

    @functools.wraps(func)
    def cached_func(*args, **kwargs):
        cache_key = key(*args, **kwargs)
        if cache_key not in cache_memory:
            cache_memory[cache_key] = func(*args, **kwargs)
        return cache_memory[cache_key]

    cached_func.cache = cache_memory

    return cached_func



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


def deprecated(func_or_class):
    if isinstance(func_or_class, type):
        class NewClass(func_or_class):
            def __init__(self, *args, **kwargs):
                warnings.warn('Deprecated class {} is instantiated.'.format(func_or_class.__name__),
                              DeprecationWarning)
                return super().__init__(*args, **kwargs)

        # from . import klass
        # klass.rename_class(NewClass, func_or_class.__name__)

        def rename_class(cls, new_name):
            cls.__name__ = new_name
            cls.__qualname__ = new_name

        rename_class(NewClass, func_or_class.__name__)

        return NewClass
    else:
        @functools.wraps(func_or_class)
        def new_func_or_class(*args, **kwargs):
            # https://docs.python.org/3/library/warnings.html
            warnings.warn('Deprecated function {} is called.'.format(func_or_class.__name__),
                          DeprecationWarning)
            return func_or_class(*args, **kwargs)

        return new_func_or_class


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


def notimplemented(func):
    @functools.wraps(func)
    def new_func(*args, **kwargs):
        raise NotImplementedError(f'The function "{func.__name__}" is not implemented')

    return new_func


def prohibit(func):
    @functools.wraps(func)
    def new_func(*args, **kwargs):
        raise Exception(f'The function "{func.__name__}" is prohibited')

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


FILE_CACHE_DEFAULT_FORMAT = 'extended_json_pretty'


def file_cache(file_path_arg_name='file_cache_path', *, save_fn=None, load_fn=None, format=None):
    '''
    Example 1
    >>> from dhnamlib.pylib.filesys import json_save, json_load
    >>> from dhnamlib.pylib.decoration import file_cache
    >>>
    >>> @file_cache('cache_path', save_fn=json_save, load_fn=json_load)
    ... def make_dict_and_print(*pairs):
    ...     d = dict(pairs)
    ...     print('make_dict_and_print is called')
    ...     return d
    >>>
    >>> pairs = [['a', 2], ['b', 4], ['c', 6]]                                 # doctest: +SKIP
    >>> d1 = make_dict_and_print(*pairs, cache_path='./path/to/cache.json')    # doctest: +SKIP
    >>> d2 = make_dict_and_print(*pairs, cache_path='./path/to/cache.json')    # doctest: +SKIP
    >>> print(d1 == d2)                                                        # doctest: +SKIP

    Example 2
    >>> @file_cache(format='json')
    ... def make_dict_and_print(*pairs):
    ...     d = dict(pairs)
    ...     print('make_dict_and_print is called')
    ...     return d
    '''
    import os

    if format is save_fn is load_fn is None:
        format = FILE_CACHE_DEFAULT_FORMAT

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
                filesys.mkpdirs_unless_exist(file_path)
                save_fn(obj, file_path)
            return obj

        return file_cache_func

    return file_cache_decorator


fcache = file_cache(format=FILE_CACHE_DEFAULT_FORMAT)


# Register
class Register:
    '''
    Example
    >>> register = Register(strategy='lazy')
    >>> name_fn = register.retrieve('name-fn')
    >>>
    >>> @register('name-fn')
    ... def full_name(first, last):
    ...     return ' '.join([first, last])
    >>>
    >>> print(name_fn('John', 'Smith'))
    John Smith
    '''

    STRATEGIES = ('instant', 'lazy', 'conditional')

    def __init__(self, strategy='instant'):
        assert strategy in self.STRATEGIES
        self.strategy = strategy
        self.memory = dict()

    def _normalize_identifier(self, identifier):
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
            registered = LazyRegisterValue(self, identifier)
        else:
            assert strategy in ['instant', 'conditional']
            registered = self.memory[identifier]

        return registered

    @staticmethod
    def _msg_not_registered(identifier):
        return f'"{identifier}" is not registered.'

    def update(self, pairs, **kwargs):
        _pairs = pairs.memory.items() if isinstance(pairs, Register) else pairs
        self.memory.update(_pairs, **kwargs)

    def items(self):
        return self.memory.items()


class MethodRegister(Register):
    '''
    Example:

    >>> class User:
    ...     method_register = MethodRegister()
    ...
    ...     def __init__(self, first_name, last_name):
    ...         self.first_name = first_name
    ...         self.last_name = last_name
    ...         self.register = self.method_register.instantiate(self)
    ...
    ...     @method_register('full_name')
    ...     def get_full_name(self):
    ...         return f'{self.first_name} {self.last_name}'
    ...
    ...     @method_register('id')
    ...     def get_id(self):
    ...         return f'{self.first_name}-{self.last_name}'.lower()
    ...
    ...     def get(self, key):
    ...         return self.register.retrieve(key)()

    >>> user = User('John', 'Smith')
    >>> user.get('id')
    'john-smith'
    '''

    def instantiate(self, obj):
        register = Register()
        for key, value in self.items():
            if hasattr(obj, value.__name__):
                register(key, getattr(obj, value.__name__))
            else:
                register(key, value)
        return register


class LazyRegisterValue:
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
def construct(construct_fn, func, /, from_kwargs=False):
    '''
    Example

    >>> @construct(dict)
    ... def make_dict(key_seq, value_fn):
    ...     for key in key_seq:
    ...         yield key, value_fn(key)

    >>> make_dict(range(5), str)
    {0: '0', 1: '1', 2: '2', 3: '3', 4: '4'}
    '''

    if from_kwargs:
        @functools.wraps(func)
        def decorated_func(*args, **kwargs):
            return construct_fn(**dict(func(*args, **kwargs)))
    else:
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


@curry
def running(func, no_return=True):
    '''
    >>> @running
    ... def print_hello():
    ...    print('hello')
    hello
    '''
    result = func()
    if no_return:
        result is None
    return func


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


@deprecated
class _handle_exception(ContextDecorator):
    '''
    A decorator for exception handling.

    Example:

    >>> @_handle_exception((TypeError, ValueError), lambda e: print(e))
    ... @_handle_exception(ZeroDivisionError, lambda e: print('oops'))
    ... def bad_idea(x):
    ...     return 1/x

    >>> bad_idea(0)  # oops
    oops
    >>> bad_idea('spam')  # unsupported operand type(s) for /: 'int' and 'str'
    unsupported operand type(s) for /: 'int' and 'str'
    >>> bad_idea(1)  # 1.0
    1.0

    This code is copied from `here <https://hissp.readthedocs.io/en/v0.2.0/faq.html#but-i-need-to-handle-the-exception-if-and-only-if-it-was-raised-for-multiple-exception-types-or-i-need-to-get-the-exception-object>`_ .
    '''

    def __init__(self, catch, handler):
        self.catch = catch
        self.handler = handler

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exception, traceback):
        if isinstance(exception, self.catch):
            handler_return = self.handler(exception)
            assert handler_return is None

            # Return True to suppress the exception
            # https://stackoverflow.com/a/43946444
            return True


def excepting(exception, default_fn=NO_VALUE, default_value=NO_VALUE):
    '''
    A decorator for exception handling.

    Example:

    >>> @excepting((TypeError, ValueError), default_fn=lambda *args, **kwargs: 'TypeError or ValueError occured')
    ... @excepting(ZeroDivisionError, default_fn=lambda *args, **kwargs: 'ZeroDivisionError occured')
    ... def bad_idea(x):
    ...     return 1/x

    >>> bad_idea(0)
    'ZeroDivisionError occured'
    >>> bad_idea('spam')
    'TypeError or ValueError occured'
    >>> bad_idea(1)
    1.0
    '''

    if default_fn is NO_VALUE:
        assert default_value is not NO_VALUE

        def default_fn(*args, **kwargs):
            return default_value
    else:
        assert default_value is NO_VALUE

    def decorator(func):
        @functools.wraps(func)
        def decorated(*args, **kwargs):
            try:
                result = func(*args, **kwargs)
            except exception:
                result = default_fn(*args, **kwargs)
            return result

        return decorated

    return decorator


def to_variables(func):
    '''
    Make a function whose result is assigned to one variable or more variables.

    Example:

    >>> @to_variables
    ... def square(*args):
    ...     return [arg ** 2 for arg in args]

    >>> a, b, c = square(1, 2, 3)
    >>> d = square(4)
    >>> print(a, b, c, d)
    1 4 9 16
    '''

    @functools.wraps(func)
    def decorated(*args, **kwargs):
        return_values = func(*args, **kwargs)
        if len(return_values) == 1:
            return return_values[0]
        else:
            return return_values
    return decorated


_VALUE_PLACEHOLDER = object()


def attr2str(obj=NO_VALUE):
    '''
    Example:

    >>> @attr2str
    ... class Fruit:
    ...     APPLE = attr2str()
    ...     BANANA = attr2str()
    ...     ORANGE = attr2str()

    >>> [Fruit.APPLE, Fruit.BANANA, Fruit.ORANGE]
    ['APPLE', 'BANANA', 'ORANGE']
    '''
    if obj is NO_VALUE:
        return _VALUE_PLACEHOLDER
    else:
        for attr, value in vars(obj).items():
            if value is _VALUE_PLACEHOLDER:
                setattr(obj, attr, attr)
        return obj


def idhashing(cls_or_obj):
    """
    >>> @idhashing
    ... class IDHashingDict(dict):
    ...     pass

    >>> d1 = IDHashingDict(a=10, b=20)
    >>> d2 = IDHashingDict(c=30, d=40)
    >>> dict([[d1, 100], [d2, 200]])
    {{'a': 10, 'b': 20}: 100, {'c': 30, 'd': 40}: 200}

    >>> original_dic = dict(a=10, b=20)
    >>> wrapped_dic = idhashing(original_dic)
    >>> wrapped_dic
    IDHashing({'a': 10, 'b': 20})

    >>> wrapped_dic['a']
    10

    >>> dict([[wrapped_dic, 300,], ['other-key', 400]])
    {IDHashing({'a': 10, 'b': 20}): 300, 'other-key': 400}
    """
    if isinstance(cls_or_obj, type):
        return _cls_idhashing(cls_or_obj)
    else:
        return _IDHashing(cls_or_obj)


def _cls_idhashing(cls):
    """
    >>> @_cls_idhashing
    ... class IDHashingDict(dict):
    ...     pass

    >>> d1 = IDHashingDict(a=10, b=20)
    >>> d2 = IDHashingDict(c=30, d=40)
    >>> sorted(repr(d) for d in set([d1, d2]))
    ["{'a': 10, 'b': 20}", "{'c': 30, 'd': 40}"]
    """

    # https://stackoverflow.com/a/4901847
    # https://stackoverflow.com/a/2909119

    def __hash__(self):
        # object.__hash__ divides id(some_object) by 16
        # https://stackoverflow.com/a/11324771
        return id(self) // 16

    def __eq__(self, other):
        return id(self) == id(other)

    def __ne__(self, other):
        return not (self == other)

    # def __lt__(self, other):
    #     return id(self) < id(other)

    cls.__hash__ = __hash__
    cls.__eq__ = __eq__
    cls.__ne__ = __ne__
    # cls.__lt__ = __lt__

    return cls


class _IDHashing:
    """
    >>> dic1 = dict(a=10, b=20)
    >>> dic2 = _IDHashing(dic1)
    >>> dic2
    IDHashing({'a': 10, 'b': 20})
    >>> dic2['a']
    10
    """

    def __init__(self, obj):
        self._original_obj = obj
        for attr in dir(obj):
            # if (
            #         attr not in self.__overridden_attrs and
            #         attr not in self.__unassignable_attrs
            # ):
            #     setattr(self, attr, getattr(obj, attr))
            if not (attr.startswith('__') and attr.endswith('__')):
                setattr(self, attr, getattr(obj, attr))

    # https://stackoverflow.com/a/4901847
    # https://stackoverflow.com/a/2909119

    def __hash__(self):
        # object.__hash__ divides id(some_object) by 16
        # https://stackoverflow.com/a/11324771
        return id(self._original_obj) // 16

    def __eq__(self, other):
        if isinstance(other, _IDHashing):
            return id(self._original_obj) == id(other._original_obj)
        else:
            return False

    def __ne__(self, other):
        return not (self == other)

    def __repr__(self):
        return type(self).__name__ + f'({repr(self._original_obj)})'

    @staticmethod
    def _clone_method(func_name):
        def clone(self, *args, **kwargs):
            return getattr(self._original_obj, func_name)(*args, **kwargs)
        return clone


for func_name in ['__setitem__', '__getitem__', '__delitem__', '__contains__', '__iter__', '__len__',
                  '__getslice__', '__setslice__']:
    setattr(_IDHashing, func_name, _IDHashing._clone_method(func_name))


_IDHashing.__name__ = 'IDHashing'
_IDHashing.__qualname__ = 'IDHashing'
