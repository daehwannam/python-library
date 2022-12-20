
import re
import itertools
from abc import abstractmethod


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


class Interface:
    "Collection of decorators for abstract classes"

    def __init__(self, *classes):
        "classes are super classes"

        self.classes = classes
        self.mro_classes = set(itertools.chain(*map(lambda cls: cls.mro(), classes)))

    def __iter__(self):
        yield from self.classes

    def implement(self, method):
        "check if the method implements an abstract method"

        # https://stackoverflow.com/a/46194542
        assert any(map(lambda cls: (hasattr(cls, "__abstractmethods__") and
                                    (method.__name__ in cls.__abstractmethods__)),
                       self.mro_classes)), \
            "'{}' is not declared as abstract method for any super class".format(method.__name__)
        return method

    def redeclare(self, method):
        "re-declare a non-implemented abstract method"

        # https://stackoverflow.com/a/46194542
        assert any(map(lambda cls: (method.__name__ in cls.__abstractmethods__), self.mro_classes)), \
            "'{}' is not declared as abstract method for any super class".format(method.__name__)
        return abstractmethod(method)

    def override(self, method):
        "check if the method overrides a method"

        # https://stackoverflow.com/a/46194542
        assert any(map(lambda cls: (method.__name__ in dir(cls)), self.classes)), \
            "'{}' does not override a method defined in super classes".format(method.__name__)
        return method


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
