
import re
import warnings
from argparse import Namespace
from enum import Enum, auto as enum_auto
from itertools import chain

from .iteration import distinct_pairs
from .exception import DuplicateValueError
from .lazy import LazyEval, eval_lazy_obj, get_eval_obj_unless_lazy
from .decoration import id_cache
from .constant import NO_VALUE
from .text import camel_to_symbol


class AttrDict(dict):
    # https://stackoverflow.com/a/14620633
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

    def rename_attr(self, old_name, new_name):
        assert new_name not in self
        self[new_name] = self[old_name]
        del self[old_name]

    # def get_attr_gen(self, *attr_keys):
    #     # if len(attr_keys) == 0:
    #     #     return ()
    #     if len(attr_keys) == 1:
    #         if isinstance(attr_keys[0], str):
    #             attr_keys = attr_keys[0].replace(',', '').split()
    #         else:
    #             attr_keys = attr_keys[0]  # attr_keys[0] is a sequence such as list or tuple
    #     for attr in attr_keys:
    #         yield self[attr]


# def get_recursive_attr_dict(obj):
#     if isinstance(obj, AttrDict):
#         return obj
#     elif isinstance(obj, dict):
#         return AttrDict(
#             (k, get_recursive_attr_dict(v))
#             for k, v in obj.items())
#     elif any(isinstance(obj, typ) for typ in [list, tuple, set]):
#         return type(obj)(get_recursive_attr_dict(elem) for elem in obj)
#     else:
#         return obj


class XNamespace(Namespace):
    """Extended Namespace

    >>> ns = XNamespace(a='A', b='B', c='C')
    >>> print(len(ns))
    3
    >>> print(ns.pop('a'))
    A
    >>> print(len(ns))
    2
    >>> print(ns.pop(b=False))
    B
    >>> print(len(ns))
    2
    >>> print(ns.pop(b=True))
    B
    >>> print(len(ns))
    1
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __bool__(self):
        return bool(self.__dict__)

    def __len__(self):
        return len(self.__dict__)

    @property
    @id_cache
    def popper(self):
        return AttrPopper(self)

    def pop(self, key=NO_VALUE, popping=True, **kwargs):
        if key is NO_VALUE:
            assert len(kwargs) == 1
            [[key, popping]] = kwargs.items()
        else:
            assert len(kwargs) == 0

        value = self.__dict__[key]
        if popping:
            del self.__dict__[key]

        return value

    def __iter__(self):
        return self.__dict__.__iter__()

    def items(self):
        return self.__dict__.items()


class AttrPopper:
    """
    >>> ns = XNamespace(a='A', b='B')
    >>> print(len(ns))
    2
    >>> print(ns.popper.a)
    A
    >>> print(len(ns))
    1
    """

    def __init__(self, namespace):
        self.namespace = namespace

    def __getattr__(self, key):
        value = getattr(self.namespace, key)
        delattr(self.namespace, key)
        return value


class LazyDict(dict):
    def __getitem__(self, key):
        return eval_lazy_obj(super().__getitem__(key))

    def get(self, key, default):
        return eval_lazy_obj(super().get(key, default))

    def items(self, lazy=True):
        if lazy:
            return super().items()
        else:
            return self._evaluated_items()

    def _evaluated_items(self):
        for key, value in super().items():
            yield key, eval_lazy_obj(value)

    def values(self):
        return map(eval_lazy_obj, super().values())


class TreeStructure:
    @classmethod
    def create_root(cls, value, terminal=False):
        struct = cls(value, terminal, None)
        struct.opened = not terminal
        return struct

    def __init__(self, value, terminal, prev):
        self.value = value
        self.terminal = terminal
        self.prev = prev

    def __repr__(self):
        lisp_style = True
        enable_prev = True

        num_reduces = 0
        tree = self
        while not tree.is_closed_root():
            num_reduces += 1
            tree = tree.reduce()

        representation = self.repr_opened(
            lisp_style=lisp_style, enable_prev=enable_prev)

        # 'representation' may have a trailing whitespace char, so '.strip' is used
        return representation.strip() + "}" * num_reduces

    def repr_opened(self, lisp_style, enable_prev, symbol_repr=False):
        representation = str(self.value)  # or repr_opened(self.value)
        if symbol_repr:
            representation = camel_to_symbol(representation)
        if not self.terminal:
            representation = '(' + representation + ' ' if lisp_style else \
                             representation + '('
            if not self.opened:
                delimiter = ' ' if lisp_style else ', '
                representation = representation + \
                    delimiter.join(child.repr_opened(lisp_style=lisp_style, enable_prev=False)
                                   for child in self.children) + ')'
        if self.prev and enable_prev:
            if self.prev.is_closed():
                delimiter = ' ' if lisp_style else ', '
                representation = '{}{}{}'. format(
                    self.prev.repr_opened(lisp_style=lisp_style, enable_prev=True),
                    delimiter, representation)
            else:
                representation = '{}{}'. format(
                    self.prev.repr_opened(lisp_style=lisp_style, enable_prev=True),
                    representation)

        return representation

    def is_opened(self):
        return not self.terminal and self.opened

    def is_closed(self):
        return self.terminal or not self.opened

    def push_term(self, value):
        return self.__class__(value, True, self)

    def push_nonterm(self, value):
        tree = self.__class__(value, False, self)
        tree.opened = True
        return tree

    def reduce(self, value=None):
        opened_tree, children = self.get_opened_tree_children()
        return opened_tree.reduce_with_children(children, value)

    def reduce_with_children(self, children, value=None):
        if value is None:
            value = self.value

        new_tree = self.__class__(value, False, self.prev)
        new_tree.opened = False
        new_tree.children = children

        return new_tree

    def get_parent_siblings(self):
        tree = self.prev  # starts from prev
        reversed_siblings = []
        while tree.is_closed():
            reversed_siblings.append(tree)
            tree = tree.prev
        return tree, tuple(reversed(reversed_siblings))

    def get_opened_tree_children(self):
        tree = self  # starts from self
        reversed_children = []
        while tree.is_closed():
            reversed_children.append(tree)
            tree = tree.prev
        return tree, tuple(reversed(reversed_children))

    def is_root(self):
        return self.prev is None

    def is_closed_root(self):
        return self.is_closed() and self.is_root()

    def is_opened_root(self):
        return self.is_opened() and self.is_root()

    def is_complete(self):  # == is_closed_root
        warnings.warn("'Warning: is_complete' is deprecated. Use 'is_closed_root' instead",
                      DeprecationWarning)
        return self.is_root() and self.is_closed()

    def get_values(self):
        values = []
        self._construct_values(values)
        return values  # values don't include it's parent

    def _construct_values(self, values):
        values.append(self.value)
        if not self.terminal:
            for child in self.children:
                child._construct_values(values)

    def get_all_values(self):
        def get_values(tree):
            if tree.is_closed():
                return tree.get_values()
            else:
                return [tree.value]

        all_values = []

        def recurse(tree):
            if not tree.is_root():
                parent, siblings = tree.get_parent_siblings()
                recurse(parent)
                for sibling in siblings:
                    all_values.extend(get_values(sibling))
            all_values.extend(get_values(tree))

        recurse(self)
        return all_values

    def get_value_tree(self):
        def recurse(tree):
            if not tree.terminal:
                return tuple(chain([tree.value], (recurse(child) for child in tree.children)))
            else:
                return tree.value

        # a value_tree doesn't include it's parent
        return recurse(self)

    def count_nodes(self, enable_prev=True):
        count = 1
        if enable_prev and self.prev:
            count += self.prev.count_nodes()
        if not self.terminal and not self.opened:
            for child in self.children:
                count += child.count_nodes(False)
        return count

    def get_last_value(self):
        if self.terminal or self.opened:
            return self.value
        else:
            return self.children[-1].get_last_value()

    def find_sub_tree(self, item, key=lambda x: x.value):
        # don't consider 'prev'
        if key(self) == item:
            return self
        elif self.terminal:
            return None
        else:
            for child in self.children:
                sub_tree = child.find_sub_tree(item, key)
                if sub_tree:
                    return sub_tree
            else:
                return None


class bidict(dict):
    '''
    Bidirectional dictionary

    Example:

    >>> dic = bidict(a=10, b=20, c=30)
    >>> dic
    {'a': 10, 'b': 20, 'c': 30}
    >>> dic.inverse
    {10: 'a', 20: 'b', 30: 'c'}
    '''

    def __init__(self, *args, **kwargs):
        try:
            super().__init__(*args, **kwargs)
            self.inverse = dict(distinct_pairs(map(reversed, self.items())))
        except DuplicateValueError:
            raise DuplicateValueError

    def __setitem__(self, key, value):
        if value in self.inverse:
            raise DuplicateValueError(
                f'tried to map a key {repr(key)} to value {repr(value)} which is already paired with key {repr(self.inverse[value])}')
        else:
            if key in self:
                self._del_inverse_item(key)
            self.inverse[value] = key
            super().__setitem__(key, value)

    def __delitem__(self, key):
        self._del_inverse_item(key)
        super().__delitem__(key)

    def _del_inverse_item(self, key):
        value = self[key]
        del self.inverse[value]


class abidict(dict):
    '''
    Asymmetry bidirectional dictionary

    Example:

    >>> dic = abidict(a=10, b=20, c=10)
    >>> dic
    {'a': 10, 'b': 20, 'c': 10}
    >>> dic.inverse              # doctest: +SKIP
    {10: {'c', 'a'}, 20: {'b'}}  # doctest: +SKIP
    '''

    def __init__(self, *args, **kwargs):
        super(abidict, self).__init__(*args, **kwargs)
        self.inverse = {}
        for key, value in self.items():
            self.inverse.setdefault(value, set()).add(key)

    def __setitem__(self, key, value):
        if key in self:
            self._del_inverse_item(key)
        super(abidict, self).__setitem__(key, value)
        self.inverse.setdefault(value, set()).add(key)

    def __delitem__(self, key):
        self._del_inverse_item(key)
        super(abidict, self).__delitem__(key)

    def _del_inverse_item(self, key):
        value = self[key]
        self.inverse[value].remove(key)
        if not self.inverse[value]:
            del self.inverse[value]


sep_pattern = re.compile('[ ,]+')


def namedlist(typename, field_names):
    '''
    Example:

    >>> A = namedlist('A', 'x,    y    z')
    >>> A
    <class 'dhnamlib.pylib.structure.A'>
    >>> a = A(1, z=3, y=2)
    >>> a
    A(x=1, y=2, z=3)
    >>> a.x
    1
    >>> a.x = 10
    >>> a[0]
    10
    >>> a
    A(x=10, y=2, z=3)
    '''

    if not isinstance(field_names, (list, tuple)):
        field_names = sep_pattern.split(field_names)
    name_to_idx_dict = dict(map(reversed, enumerate(field_names)))

    class NamedList(list):
        def __init__(self, *args, **kwargs):
            assert (len(args) + len(kwargs)) == len(field_names)
            super(NamedList, self).__init__([None] * len(field_names))
            for idx in range(len(args)):
                self[idx] = args[idx]
            for k, v in kwargs.items():
                self[name_to_idx_dict[k]] = v

        def __getattr__(self, key):
            if key in name_to_idx_dict:
                return self[name_to_idx_dict[key]]
            else:
                return super(NamedList, self).__getattr__(key)

        def __setattr__(self, key, value):
            if key in name_to_idx_dict:
                self[name_to_idx_dict[key]] = value
            else:
                # super(NamedList, self).__setattr__(key, value)
                raise AttributeError("'NamedList' object doesn't allow new attributes.")

        def __repr__(self):
            return ''.join(
                [f'{typename}',
                 '(',
                 ', '.join(f'{name}={self[name_to_idx_dict[name]]}' for name in field_names),
                 ')'])

        def append(self, *args, **kwargs):
            raise Exception('This method is not used')

        def extend(self, *args, **kwargs):
            raise Exception('This method is not used')

        @classmethod
        def get_attr_idx(cls, attr):
            return name_to_idx_dict[attr]

    NamedList.__name__ = typename
    NamedList.__qualname__ = typename
    return NamedList


class _NameEnum(Enum):
    """
    Source: https://docs.python.org/3.9/library/enum.html#using-automatic-values

    Limitation: a subclass of Enum cannot have its subclass.

    Example:

    >>> class Ordinal(_NameEnum):
    ...     auto = _NameEnum.auto
    ...
    ...     NORTH = auto()
    ...     SOUTH = auto()
    ...     EAST = auto()
    ...     WEST = auto()
    ...
    >>> list(Ordinal)
    [<Ordinal.NORTH: 'NORTH'>, <Ordinal.SOUTH: 'SOUTH'>, <Ordinal.EAST: 'EAST'>, <Ordinal.WEST: 'WEST'>]
    """
    def _generate_next_value_(name, start, count, last_values):
        return name

    @staticmethod
    def auto():
        return enum_auto()


class SetWrapper:
    """
    >>> s1 = {1, 2, 3}
    >>> s2 = SetWrapper(s1)
    >>> 3 in s2
    True
    """

    def __init__(self, _set):
        self._set = _set

    def __contains__(self, elem):
        return self._set.__contains__(elem)

    def __iter__(self):
        return self._set.__iter__()


class DictWrapper:
    """
    >>> d1 = dict(a=10, b=20)
    >>> d2 = DictWrapper(d1)
    >>> d2['a']
    10
    """

    def __init__(self, _dict):
        self._dict = _dict

    def __contains__(self, elem):
        return self._dict.__contains__(elem)

    def __iter__(self):
        return self._dict.__iter__()

    def keys(self):
        return self._dict.keys()

    def values(self):
        return self._dict.values()

    def items(self):
        return self._dict.items()

    def __getitem__(self, key):
        return self._dict.__getitem__(key)

    def get(self, key, default=None):
        return self._dict.get(key, default)
