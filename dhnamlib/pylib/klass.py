
import re
import itertools
import functools
from abc import abstractmethod
from .decoration import deprecated


def get_all_subclass_set(cls):
    subclass_list = []

    def recurse(klass):
        for subclass in klass.__subclasses__():
            subclass_list.append(subclass)
            recurse(subclass)

    recurse(cls)

    return set(subclass_list)


def find_unique_subclass_in_module(superclass, module):
    subclass = None
    for obj_name in dir(module):
        klass = getattr(module, obj_name)
        if klass != superclass and \
           isinstance(type(klass), type) and \
           isinstance(klass, type) and \
           issubclass(klass, superclass):
            if subclass is None:
                subclass = klass
            else:
                raise Exception("No more than 1 subclass of {} is allowed.".format(superclass.__name__))
    if subclass is None:
        raise Exception("No subclass of {} exists".format(superclass.__name__))

    return subclass


class AttrValidator:
    separator = re.compile(r'[ ,]+')

    def __init__(self, attrs):
        if isinstance(attrs, (tuple, list, set)):
            pass
        elif isinstance(attrs, str):
            attrs = self.separator.split(attrs)
        else:
            raise Exception('Wrong description of attributes')

        self.attrs = tuple(attrs)

    def validate(self, instance):
        for attr in self.attrs:
            assert hasattr(instance, attr), f"{instance} doesn't have '{attr}' as an attribute"


def abstractfunction(func):
    @functools.wraps(func)
    def new_func(*args, **kwargs):
        raise NotImplementedError(
            "The abstract function '{}' is not implemented".format(func.__name__))

    new_func.__isabstractfunction__ = True
    return new_func


def isabstractfunction(function):
    # https://stackoverflow.com/a/46194542
    return getattr(function, '__isabstractfunction__', False)


def isabstractmethod(method):
    # https://stackoverflow.com/a/46194542
    return getattr(method, '__isabstractmethod__', False)


@deprecated
def _implement_fn(abstract_fn):
    def decorate(concrete_fn):
        assert isabstractfunction(abstract_fn), \
            "'{}' is not declared as abstract".format(abstract_fn.__name__)
        assert abstract_fn.__name__ == concrete_fn.__name__, \
            "The function names are different: {} and {}".format(abstract_fn.__name__, concrete_fn.__name__)
        concrete_fn.__abstractfunction__ = abstract_fn
        return concrete_fn
    return decorate


_NO_FUNCTION = object()


@deprecated
def _is_implementing_fn(concrete_fn, abstract_fn):
    return getattr(concrete_fn, '__abstractfunction__', _NO_FUNCTION) is abstract_fn


@deprecated
def _get_concrete(abstract_fn, module):
    concrete_fn = getattr(module, abstract_fn.__name__)
    assert _is_implementing_fn(concrete_fn, abstract_fn), \
        f'The function {concrete_fn.__name__} is not decorated with `implement_fn`'
    return concrete_fn


class Interface:
    """Collection of decorators for abstract classes

    Example

    >>> from abc import ABCMeta, abstractmethod
    >>>
    >>> class A(metaclass=ABCMeta):
    ...     @abstractmethod
    ...     def func(self):
    ...         pass

    >>> class B(A):
    ...     interface = Interface(A)
    ...     @interface.implement
    ...     def func(self):
    ...         print('This is implemented.')
    """

    def __init__(self, *parents):
        "`parents` are super classes"

        assert len(parents) > 0

        self.parents = parents
        self.mro_classes = set(itertools.chain(*map(lambda cls: cls.mro(), parents)))

    def __iter__(self):
        yield from self.parents

    def implement(self, method):
        "check if the method implements an abstract definition"

        # https://stackoverflow.com/a/46194542

        assert self._declared_as_abstract(method), \
            "'{}' is not declared as abstract for any super class".format(method.__name__)
        assert not self._implemeted_as_abstract(method), \
            "'{}' is already implemented, so override it instead".format(method.__name__)
        return method

    def redeclare(self, method):
        "re-declare a non-implemented abstract definition"

        assert self._declared_as_abstract(method), \
            "'{}' is not declared as abstract for any super class".format(method.__name__)
        return abstractmethod(method)

    def override(self, method):
        "check if the method overrides a method"

        # assert not self.__declared_as_abstractmethod(method),

        assert self._existing_in_parents(method), \
            "'{}' does not exist in parent classes".format(method.__name__)
        assert not self._is_abstract_in_parents(method), \
            "'{}' is abstract, so implement it rather than override".format(method.__name__)
        return method

    # method_or_func
    def _declared_as_abstract(self, method_or_func):
        return self.__declared_as_abstractmethod(method_or_func) or \
            self.__declared_as_abstractfunction(method_or_func)

    def _is_abstract_in_parents(self, method_or_func):
        return self.__is_abstractmethod_in_parents(method_or_func) or \
            self.__is_abstractfunction_in_parents(method_or_func)

    def _implemeted_as_abstract(self, method_or_func):
        return self.__implemented_as_abstractmethod(method_or_func) or \
            self.__implemented_as_abstractfunction(method_or_func)

    # method
    def __declared_as_abstractmethod(self, method):
        return any(map(lambda cls: (hasattr(cls, "__abstractmethods__") and
                                    (method.__name__ in cls.__abstractmethods__)),
                       self.mro_classes))

    def __is_abstractmethod_in_parents(self, method):
        return any(map(lambda cls: (isabstractmethod(getattr(cls, method.__name__))), self.parents))

    def __implemented_as_abstractmethod(self, method):
        return self.__declared_as_abstractmethod(method) and \
            not self.__is_abstractmethod_in_parents(method)

    # function
    def __declared_as_abstractfunction(self, function):
        return any(map(lambda cls: (hasattr(cls, function.__name__) and
                                    isabstractfunction(getattr(cls, function.__name__))),
                       self.mro_classes))

    def __is_abstractfunction_in_parents(self, function):
        return any(map(lambda cls: (isabstractfunction(getattr(cls, function.__name__))), self.parents))

    def __implemented_as_abstractfunction(self, function):
        return self.__declared_as_abstractfunction(function) and \
            not self.__is_abstractfunction_in_parents(function)

    # others
    def _existing_in_parents(self, method):
        return any(map(lambda cls: (method.__name__ in dir(cls)), self.parents))


def _test_interface():
    from abc import ABCMeta, abstractmethod

    class A(metaclass=ABCMeta):
        @abstractmethod
        def func1(self):
            pass

        @abstractfunction
        def func2(self):
            pass

    class B(A):
        interface = Interface(A)

        @interface.implement
        def func1(self):
            print('This is implemented')

        @interface.implement
        def func2(self):
            print('This is implemented')

    class C(B):
        interface = Interface(B)

        @interface.override
        def func1(self):
            print('This is overriden')

        @interface.override
        def func2(self):
            print('This is overriden')


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


def abstractproperty(func):
    # It's identical with abc.abstractproperty
    return property(abstractmethod(func))


def abstractclassmethod(func):
    # https://stackoverflow.com/a/60180758
    #
    # It's identical with abc.abstractclassmethod
    return classmethod(abstractmethod(func))


# alias
abstractattribute = abstractmethod


def subclass(cls):
    """Collection of decorators for abstract classes

    Example

    >>> from abc import ABCMeta, abstractmethod
    >>>
    >>> class A(metaclass=ABCMeta):
    ...     @abstractmethod
    ...     def foo(self):
    ...         pass
    ...     def bar(self):
    ...         pass
    >>>
    >>> @subclass
    ... class B(A):
    ...     @implement
    ...     def foo(self):
    ...         print('This is implemented.')
    ...     @override
    ...     def bar(self):
    ...         pass

    """

    interface = Interface(*cls.__bases__)

    for attr, value in vars(cls).items():
        if hasattr(value, '_implement'):
            delattr(value, '_implement')
            setattr(cls, attr, interface.implement(value))
        elif hasattr(value, '_redeclare'):
            delattr(value, '_redeclare')
            setattr(cls, attr, interface.redeclare(value))
        elif hasattr(value, '_override'):
            delattr(value, '_override')
            setattr(cls, attr, interface.override(value))

    return cls


def implement(func):
    func._implement = True
    return func


def redeclare(func):
    func._redeclare = True
    return func


def override(func):
    func._override = True
    return func
