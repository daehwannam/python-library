
import functools
import itertools
from abc import abstractmethod


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


# https://stackoverflow.com/a/5191224
class ClassPropertyDescriptor:
    def __init__(self, fget, fset=None):
        self.fget = fget
        self.fset = fset

    def __get__(self, obj, klass=None):
        if klass is None:
            klass = type(obj)
        return self.fget.__get__(obj, klass)()

    def __set__(self, obj, value):
        if not self.fset:
            raise AttributeError("can't set attribute")
        type_ = type(obj)
        return self.fset.__get__(obj, type_)(value)

    def setter(self, func):
        if not isinstance(func, (classmethod, staticmethod)):
            func = classmethod(func)
        self.fset = func
        return self


def classproperty(func):
    if not isinstance(func, (classmethod, staticmethod)):
        func = classmethod(func)

    return ClassPropertyDescriptor(func)


def abstractfunction(func):
    @functools.wraps(func)
    def new_func(*args, **kargs):
        raise NotImplementedError(
            "The abstract function '{}' is not implemented".format(func.__name__))
    return new_func


def implement(*classes):
    "classes are super classes"

    mro_classes = set(itertools.chain(*map(lambda cls: cls.mro(), classes)))

    def implement_classes(method):
        "check if the method implements an abstract method"

        # https://stackoverflow.com/a/46194542
        assert any(map(lambda cls: (hasattr(cls, "__abstractmethods__") and
                                    (method.__name__ in cls.__abstractmethods__)),
                       mro_classes)), \
            "'{}' is not declared as abstract method for any super class".format(method.__name__)
        return method

    return implement_classes


def non_implemented_classes(*classes):
    "classes are super classes"

    mro_classes = set(itertools.chain(*map(lambda cls: cls.mro(), classes)))

    def non_implement_classes(method):
        "re-declare a non-implemented abstract method"

        # https://stackoverflow.com/a/46194542
        assert any(map(lambda cls: (method.__name__ in cls.__abstractmethods__), mro_classes)), \
            "'{}' is not declared as abstract method for any super class".format(method.__name__)
        return abstractmethod(method)

    return non_implement_classes


def overrides(*classes):
    "classes are super classes"

    def _overrides(method):
        "check if the method overrides an abstract method"

        # https://stackoverflow.com/a/46194542
        assert any(map(lambda cls: (method.__name__ in dir(cls)), classes)), \
            "'{}' does not overrides a method defined in super classes".format(method.__name__)
        return method

    return _overrides


def abstract_property(func):
    return property(abstractmethod(func))



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
