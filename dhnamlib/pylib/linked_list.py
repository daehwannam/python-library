
import types
import functools
from functools import reduce
import itertools
from . import iteration as dhnam_iter

# from .decoration import classproperty

def llist(seq=None):
    """
    Create LeftwardList
    >>> llist()
    ()
    >>> llist([0, 1, 2, 3, 4])
    (0, 1, 2, 3, 4)
    """
    if seq is None:
        return LeftwardList.create()
    else:
        return LeftwardList.from_seq(seq)


def rlist(seq=None):
    """
    Create RightwardList
    >>> rlist()
    ()
    >>> rlist([0, 1, 2, 3, 4])
    (0, 1, 2, 3, 4)
    """
    if seq is None:
        return RightwardList.create()
    else:
        return RightwardList.from_seq(seq)


def init_linked_list_class(cls):
    cls.klass = cls

    class NilList(cls):
        def __repr__(self):
            return '()'

    cls.nil = NilList()

    return cls


@init_linked_list_class
class LinkedList(tuple):
    """
    >>> from dhnamlib.pylib.linked_list import *
    >>> ll = LinkedList.create(0, 1, 2, 3, 4)
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

    # def __new__(cls, args=()):
    #     return super(LinkedList, cls).__new__(cls, tuple(args))

    @classmethod
    def create(cls, *args, leftward=True):
        return reduce(lambda lst, el: cls((el, lst)),
                      reversed(args) if leftward else args, cls.klass.nil)

    @classmethod
    def from_seq(cls, seq, leftward=True):
        if leftward:
            if isinstance(seq, types.GeneratorType):
                seq = tuple(seq)
            seq = reversed(seq)
        return reduce(lambda lst, el: cls((el, lst)), seq, cls.klass.nil)

    def cons(self, el):
        return self.klass((el, self))

    construct = cons

    def car(self):
        return super(LinkedList, self).__getitem__(0) if self.__bool__() else self

    # first = car

    def cdr(self):
        return super(LinkedList, self).__getitem__(1) if self.__bool__() else self

    rest = cdr

    def __bool__(self):
        return self is not self.klass.nil

    def null(self):
        return self is self.klass.nil

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
class LeftwardList(LinkedList):
    """
    >>> ll = LeftwardList.create(0, 1, 2, 3, 4)
    >>> ll.first()
    0
    >>> ll.rest()
    (1, 2, 3, 4)
    >>> list(ll)
    [0, 1, 2, 3, 4]
    """

    first = LinkedList.car


@init_linked_list_class
class RightwardList(LinkedList):
    """
    >>> rl = RightwardList.create(0, 1, 2, 3, 4)
    >>> rl.last()
    4
    >>> rl.rest()
    (0, 1, 2, 3)
    >>> list(rl)
    [0, 1, 2, 3, 4]
    """
    @classmethod
    def create(cls, *args, leftward=False):
        return super().create(*args, leftward=leftward)

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
class AssociationList(LinkedList):
    # @classmethod
    # def create(cls, *args, **kargs):
    #     return super().create(*itertools.chain(args, kargs.items()))

    @classmethod
    def from_pairs(cls, *args, **kargs):
        return super().create(*map(tuple, itertools.chain(args, kargs.items())))

    def get(self, attr, key=lambda x: x, defaultvalue=None, defaultfunc=None, no_default=False):
        assert sum(arg is not None for arg in [defaultvalue, defaultfunc]) <= 1

        pair = self.find(attr, key=lambda pair: key(pair[0])).car()
        if pair.null():
            if no_default:
                raise Exception("Cannot find the correspodning key")
            elif defaultfunc is not None:
                return defaultfunc()
            else:
                return defaultvalue
        return pair[1]

    def __getitem__(self, key):
        return self.get(key, no_default=True)

    def get_values(self, attr_list, key=lambda x: x, defaultvalue=None, defaultfunc=None, no_default=False):
        return dhnam_iter.get_values_from_pairs(
            self, attr_list, key, defaultvalue, defaultfunc, no_default)

    def update(self, attr, value):
        return self.construct((attr, value))

    def compact(self, key=lambda x: x):
        def get_gen():
            attr_keys = set()
            for attr, val in self:
                attr_key = key(attr)
                if attr_key not in attr_keys:
                    attr_keys.add(attr_key)
                    yield attr, val

        return self.klass.create(*get_gen())

    @staticmethod
    def compact_assoc(key):
        def compact_assoc_with_key(func):
            @functools.wraps(func)
            def compacted_func(self, *args, **kwargs):
                return func(self, *args, **kwargs).compact(key=key)
        return compact_assoc_with_key

    def keys(self):
        for key, value in self:
            yield key


# @init_linked_list_class
# class AttrList(AssociationList):

#     def __getattr__(self, attr):
#         # https://stackoverflow.com/a/5021467
#         return self.get(attr)
