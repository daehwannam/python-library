
import types
from typing import Iterable, TypeVar, Generic, Sized, Tuple
import functools
from functools import reduce
import itertools
from .iteration import get_values_from_pairs
from .decoration import deprecated

# from .decoration import classproperty

T = TypeVar('T')                # Type

KT = TypeVar('KT')              # Key Type
VT = TypeVar('VT')              # Value Type


def llist(seq: Iterable[T] = None) -> 'LeftwardList[T]':
    """
    Create LeftwardList
    >>> llist()
    ()
    >>> llist([0, 1, 2, 3, 4])
    (0, 1, 2, 3, 4)
    """
    ...


def rlist(seq: Iterable[T] = None) -> 'RightwardList[T]':
    """
    Create RightwardList
    >>> rlist()
    ()
    >>> rlist([0, 1, 2, 3, 4])
    (0, 1, 2, 3, 4)
    """
    ...


def alist(pairs: Iterable[Tuple[KT, VT]] = None, **kwargs) -> 'AssociationList[KT, VT]':
    """
    Create AssociationList
    >>> alist()
    ()
    >>> alist([['a', 10], ['b', 20]], c=30, d=40)
    (('a', 10), ('b', 20), ('c', 30), ('d', 40))
    """
    ...


def init_linked_list_class(cls):
    cls._klass = cls

    class NilList(cls):
        def __repr__(self):
            return '()'

    cls.nil = NilList(_private=_PRIVATE)

    def NilList__init__(self, *args, **kwargs):
        raise Exception('NilList.__init__ should not be called.')

    NilList.__init__ = NilList__init__

    return cls


_PRIVATE = object()

@init_linked_list_class
class LinkedList(tuple, Generic[T], Iterable[T], Sized):
    """
    >>> from dhnamlib.pylib.linked_list import *
    >>> ll = LinkedList.from_elems(0, 1, 2, 3, 4)
    >>> ll
    (0, 1, 2, 3, 4)
    >>> ll.car()
    0
    >>> ll.cdr()
    (1, 2, 3, 4)
    >>> list(ll)
    [0, 1, 2, 3, 4]
    """
    # https://stackoverflow.com/a/283630

    def __new__(cls, args=(), _private=None):
        assert _private is _PRIVATE, \
            f'{cls.__name__}.__init__ is a private method. Use {cls.__name__}.create instead.'
        return tuple.__new__(cls, tuple(args))

    @classmethod
    def create(cls, seq=None, leftward=True):
        if seq is None:
            # return cls.from_elems()
            return cls.nil
        else:
            return cls.from_seq(seq, leftward=leftward)

    @classmethod
    def from_elems(cls, *args, leftward=True):
        return reduce(lambda lst, el: cls.cons(lst, el),
                      reversed(args) if leftward else args, cls.nil)

    @classmethod
    def from_seq(cls, seq, leftward=True):
        if leftward:
            if isinstance(seq, types.GeneratorType):
                seq = tuple(seq)
            seq = reversed(seq)
        return reduce(lambda lst, el: cls.cons(lst, el), seq, cls.nil)

    def cons(self, el):
        # return self.__class__((el, self))
        return self._klass((el, self), _private=_PRIVATE)  # '_klass' is used to prevent the use of NilList.__init__

    # construct = cons

    def car(self):
        return super(LinkedList, self).__getitem__(0) if self.__bool__() else self

    # first = car

    def cdr(self):
        return super(LinkedList, self).__getitem__(1) if self.__bool__() else self

    rest = cdr

    def __bool__(self):
        return self is not self.__class__.nil

    def null(self):
        return self is self.__class__.nil

    def decons(self):
        return self.car(), self.cdr()

    def nth(self, n):
        lst = self
        while n > 0:
            n -= 1
            lst = self.cdr()
        return lst.car()

    def __len__(self):
        lst = self
        count = 0
        while lst.__bool__():
            count += 1
            lst = lst.cdr()
        return count

    def __iter__(self):
        lst = self
        while lst.__bool__():
            yield lst.car()
            lst = lst.cdr()

    def __reversed__(self):
        items = tuple(self)
        for item in reversed(items):
            yield item

    def __repr__(self):
        return str(tuple(self))

    def find(self, value, key=lambda x: x):
        lst = self
        while lst.__bool__():
            if key(lst.car()) == value:
                return lst
            lst = lst.cdr()
        return lst
        # if self.null() or key(self.car()) == value:
        #     return self
        # else:
        #     return self.cdr().find(value, key)


@init_linked_list_class
class LeftwardList(LinkedList[T], Generic[T]):
    """
    >>> llist()
    ()

    >>> ll = llist([0, 1, 2, 3, 4])
    >>> ll
    (0, 1, 2, 3, 4)

    >>> ll.first()
    0
    >>> ll.rest()
    (1, 2, 3, 4)
    >>> list(ll)
    [0, 1, 2, 3, 4]
    """

    first = LinkedList.car


@init_linked_list_class
class RightwardList(LinkedList[T], Generic[T]):
    """
    >>> rlist()
    ()

    >>> rl = rlist([0, 1, 2, 3, 4])
    >>> rl
    (0, 1, 2, 3, 4)

    >>> rl.last()
    4
    >>> rl.rest()
    (0, 1, 2, 3)
    >>> list(rl)
    [0, 1, 2, 3, 4]
    """

    @classmethod
    def create(cls, seq=None, leftward=False):
        return super().create(seq, leftward=leftward)

    @classmethod
    def from_elems(cls, *args, leftward=False):
        return super().from_elems(*args, leftward=leftward)

    @classmethod
    def from_seq(cls, seq, leftward=False):
        return super().from_seq(seq, leftward=leftward)

    def __iter__(self):
        items = tuple(self.__reversed__())
        for item in reversed(items):
            yield item

    def __reversed__(self):
        lst = self
        while lst.__bool__():
            yield lst.car()
            lst = lst.cdr()

    # def first(self):
    #     raise Exception('"first" is not used for "RightwardList". Use "last" instead')

    last = LinkedList.car


@init_linked_list_class
class AssociationList(LinkedList[Tuple[KT, VT]], Generic[KT, VT]):
    """
    >>> alist()
    ()
    >>> alist([['a', 10], ['b', 20]], c=30, d=40)
    (('a', 10), ('b', 20), ('c', 30), ('d', 40))

    >>> al = alist([['a', 10], ['b', 20], ['a', 25]], c=30, d=40)
    >>> al
    (('a', 10), ('b', 20), ('a', 25), ('c', 30), ('d', 40))
    >>> al.get('a')
    10
    >>> al.get('x') is None
    True
    >>> al['b']
    20
    >>> al.compact()
    (('a', 10), ('b', 20), ('c', 30), ('d', 40))
    """
    @classmethod
    def from_pairs(cls, pairs=None, **kwargs):
        if pairs is None:
            pairs = []
        return super().from_elems(*map(tuple, itertools.chain(pairs, kwargs.items())))

    def assoc(self, attr_key, key=lambda x: x, defaultvalue=None, defaultfunc=None, no_default=False):
        assert sum(arg is not None for arg in [defaultvalue, defaultfunc]) <= 1

        pair = self.find(attr_key, key=lambda pair: key(pair[0])).car()
        if pair is self.nil:
            if no_default:
                raise Exception("Cannot find the correspodning key")
            elif defaultfunc is not None:
                pair = attr_key, defaultfunc()
            else:
                pair = attr_key, defaultvalue
        return pair

    def get(self, attr_key, key=lambda x: x, defaultvalue=None, defaultfunc=None, no_default=False):
        _attr_key, value = self.assoc(
            attr_key, key=key, defaultvalue=defaultvalue, defaultfunc=defaultfunc, no_default=no_default)
        return value

    def __getitem__(self, key):
        return self.get(key, no_default=True)

    def get_values(self, attr_list, key=lambda x: x, defaultvalue=None, defaultfunc=None, no_default=False):
        return get_values_from_pairs(
            self, attr_list, key, defaultvalue, defaultfunc, no_default)

    def update(self, attr_key, value):
        return self.cons((attr_key, value))

    def compact(self, key=lambda x: x):
        def get_gen():
            attr_keys = set()
            for attr_key, val in self:
                attr_key = key(attr_key)
                if attr_key not in attr_keys:
                    attr_keys.add(attr_key)
                    yield attr_key, val

        return self.__class__.from_elems(*get_gen())

    @deprecated
    @staticmethod
    def make_compact_func_with_key(func, key):
        @functools.wraps(func)
        def compact_func(self, *args, **kwargs):
            return func(self, *args, **kwargs).compact(key=key)

    def keys(self):
        for key, value in self:
            yield key


# @init_linked_list_class
# class AttrList(AssociationList):

#     def __getattr__(self, attr):
#         # https://stackoverflow.com/a/5021467
#         return self.get(attr)

llist = LeftwardList.create
rlist = RightwardList.create
alist = AssociationList.from_pairs
