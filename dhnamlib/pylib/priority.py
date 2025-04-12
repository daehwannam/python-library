
import heapq
import math
from itertools import chain
from collections import deque
from dataclasses import dataclass
import typing

from .constant import NO_VALUE
from .doubly_linked_list import DoublyLinkedList, DoublyLinkedListNode


class HeapPQ:
    """
    Priority Queue using Heap.

    A unit has a form of (priority, item_num, item).
    The "item_num" is necessary not to compare values of "item" when two units have the same "priority".
    In addition, the an item that is pushed first is also popped out first if the items have the same priority.

    >>> pq = HeapPQ([[0, 'a'],
    ...              [1, 'b'],
    ...              [-1, 'c'],
    ...              [2, 'd']])
    >>> pq.heap
    [(-1, 2, 'c'), (1, 1, 'b'), (0, 0, 'a'), (2, 3, 'd')]
    >>> pq.pop()
    'c'
    >>> pq.pop()
    'a'
    >>> pq.push(0, 'e')
    >>> pq.pushpop(1, 'd')
    'e'
    >>> pq.pushpop(-2, 'x')
    'x'
    >>> pq.pop()
    'b'
    """

    def __init__(self, pairs=[]):
        self.total_item_num = 0
        self.heap = list((priority, item_num, item)
                         for item_num, (priority, item) in
                         enumerate(pairs, self.total_item_num))
        self.total_item_num = len(self.heap)
        heapq.heapify(self.heap)

    def push(self, priority, item):
        heapq.heappush(self.heap, (priority, self.total_item_num, item))
        self.total_item_num += 1

    def pop(self):
        "Return an item of the smallest value of priority"

        priority, item_num, item = heapq.heappop(self.heap)
        return item

    def pushpop(self, priority, item):
        if priority > self.root_priority:
            popped_priority, popped_item_num, popped_item = heapq.heappushpop(self.heap, (priority, self.total_item_num, item))
            self.total_item_num += 1
            return popped_item
        else:
            return item

    @property
    def root(self):
        priority, item_num, item = self.heap[0]
        return item

    @property
    def root_priority(self):
        priority, item_num, item = self.heap[0]
        return priority

    # def prune(self):
    #     pass

    def __bool__(self):
        return bool(self.heap)

    def __iter__(self):
        for priority, item_num, item in self.heap:
            yield item

    def __len__(self):
        return len(self.heap)

    def __repr__(self):
        return repr(list(self))


class PriorityDict:
    """
    >>> pdict = PriorityDict([[0, 'a', 'A'],
    ...                       [-1, 'b', 'B'],
    ...                       [-2, 'c', 'C'],
    ...                       [1, 'd', 'D'],
    ...                       [2, 'e', 'E'],
    ...                       [-3, 'f', 'F']])
    >>> pdict
    {'f': 'F', 'b': 'B', 'c': 'C', 'd': 'D', 'e': 'E', 'a': 'A'}
    >>> pdict.pop()
    ('f', 'F')
    >>> pdict.pop()
    ('c', 'C')
    >>> pdict.update(1, 'g', 'G')
    >>> pdict
    {'b': 'B', 'd': 'D', 'e': 'E', 'a': 'A', 'g': 'G'}
    >>> pdict.pop()
    ('b', 'B')
    >>> pdict.pop()
    ('a', 'A')
    >>> pdict.pop()
    ('d', 'D')
    >>> pdict
    {'e': 'E', 'g': 'G'}
    >>> pdict['e']
    'E'
    >>> pdict
    {'e': 'E', 'g': 'G'}
    """

    KEY_INDEX = 0
    VALUE_INDEX = 1
    VALID_INDEX = 2

    def __init__(self, tuples=[]):
        keys = set()
        for priority, key, value in tuples:
            if key in keys:
                raise Exception(f'Overlapping key "{key}".')
            keys.add(key)

        self.pq = HeapPQ((priority, [key, value, True])
                         for priority, key, value in tuples)
        key_index = self.KEY_INDEX
        self.unit_dict = dict([unit[key_index], unit] for unit in self.pq)

    def update(self, priority, key, value):
        """
        Update a key-value pair.

        If a "key" is alread in self.unit_dict,
        the size of self.unit_dict is unchanged.
        """

        unit = [key, value, True]
        self.pq.push(priority, unit)

        existing_unit = self.unit_dict.get(key, NO_VALUE)
        if existing_unit is not NO_VALUE:
            existing_unit[self.VALID_INDEX] = False
        self.unit_dict[key] = unit

    def pop(self):
        unit = self.pq.pop()
        while not unit[self.VALID_INDEX]:
            unit = self.pq.pop()
        del self.unit_dict[unit[self.KEY_INDEX]]
        key, value, valid = unit
        return key, value

    def __iter__(self):
        return self.keys()

    def keys(self):
        return self.unit_dict.keys()

    def items(self):
        value_index = self.VALUE_INDEX
        for key, unit in self.unit_dict.items():
            yield key, unit[value_index]

    def values(self):
        value_index = self.VALUE_INDEX
        for unit in self.unit_dict.values():
            yield unit[value_index]

    def __getitem__(self, key):
        return self.unit_dict[key][self.VALUE_INDEX]

    def __contains__(self, key):
        return key in self.unit_dict

    def __repr__(self):
        return repr(dict(self.items()))

    def get(self, key, default=None):
        return self.unit_dict.get(key, default)


class PriorityCache:
    """
    Keep itmes of largest values of priorities.

    >>> pcache = PriorityCache([[0, 'a', 'A'],
    ...                        [-1, 'b', 'B'],
    ...                        [-2, 'c', 'C'],
    ...                        [1, 'd', 'D'],
    ...                        [2, 'e', 'E'],
    ...                        [-3, 'f', 'F']],
    ...                        size=5
    ... )
    >>> pcache
    {'a': 'A', 'b': 'B', 'c': 'C', 'd': 'D', 'e': 'E'}
    >>> pcache.put(3, 'g', 'G')
    >>> pcache.put(4, 'h', 'H')
    >>> pcache.put(-3, 'i', 'I')
    >>> pcache
    {'a': 'A', 'd': 'D', 'e': 'E', 'g': 'G', 'h': 'H'}
    """
    KEY_INDEX = 0
    VALUE_INDEX = 1

    def __init__(self, tuples=[], size=float('inf')):
        self.size = size
        self.pq = HeapPQ()
        self.cache = dict()

        for priority, key, value in tuples:
            self.put(priority, key, value)

    def put(self, priority, key, value):
        """
        Add a key-value pair if the new pair has a high priority or the cache is not full.
        """

        if len(self.pq) < self.size:
            self._push(priority, key, value)
            # return None, None
        else:
            assert len(self.pq) == self.size
            # return self._pushpop(priority, key, value)
            self._pushpop(priority, key, value)

    def _push(self, priority, key, value):
        """
        Add a key-value pair.
        """

        assert key not in self.cache, f'The key "{key}" is already cached.'

        unit = [key, value]
        self.pq.push(priority, unit)
        self.cache[key] = value

    def pop(self):
        unit = self.pq.pop()
        del self.cache[unit[self.KEY_INDEX]]
        return unit

    def _pushpop(self, priority, key, value):
        assert key not in self.cache, f'The key "{key}" is already cached.'

        if priority > self.pq.root_priority:
            unit = [key, value]
            popped_unit = self.pq.pushpop(priority, unit)
            assert popped_unit is not unit
            self.cache[key] = value

            popped_key, popped_value = popped_unit
            del self.cache[popped_key]

            return popped_unit
        else:
            return key, value

    def __iter__(self):
        return self.keys()

    def keys(self):
        return self.cache.keys()

    def items(self):
        return self.cache.items()

    def values(self):
        return self.cache.values()

    def __getitem__(self, key):
        return self.cache[key]

    def __contains__(self, key):
        return key in self.cache

    def __repr__(self):
        return repr(self.cache)

    def get(self, key, default=None):
        return self.cache.get(key, default)

    def __len__(self):
        assert len(self.cache) == len(self.pq)
        return len(self.cache)


class FIFODict:
    '''
    Keep most recently updated key-value pairs

    Example:

    >>> dic = FIFODict(3)
    >>> dic
    FIFODict(3, {})
    >>> dic['a'] = 1
    >>> dic['b'] = 2
    >>> dic['c'] = 3
    >>> dic
    FIFODict(3, {'a': 1, 'b': 2, 'c': 3})
    >>> dic['a'] = 4
    >>> dic
    FIFODict(3, {'a': 4, 'b': 2, 'c': 3})
    >>> dic['d'] = 5
    >>> dic
    FIFODict(3, {'a': 4, 'c': 3, 'd': 5})
    >>> dic['e'] = 5
    >>> dic
    FIFODict(3, {'a': 4, 'd': 5, 'e': 5})
    >>> dic['a']
    4
    >>> dic.get('a')
    4
    >>> dic.get('c') is None
    True
    '''

    # Unit = namedlist('Unit', 'key, value, valid')

    KEY_INDEX = 0
    VALUE_INDEX = 1
    VALID_INDEX = 2

    def __init__(self, max_size):
        assert max_size > 0
        self.q = deque()
        self.unit_dict = {}
        self.max_size = max_size
        self.size = 0

    def _update_kv(self, key, value):
        if key in self.unit_dict:
            # old unit is no more valid
            self.unit_dict[key][self.VALID_INDEX] = False
        elif self.size < self.max_size:
            self.size += 1
        else:
            lr_unit = self.q.popleft()
            while not lr_unit[self.VALID_INDEX]:
                lr_unit = self.q.popleft()
            # Remove a valid unit from unit_dict
            del self.unit_dict[lr_unit[self.KEY_INDEX]]

        unit = [key, value, True]
        self.q.append(unit)
        self.unit_dict[key] = unit

        assert len(self.unit_dict) <= self.max_size

    def __iter__(self):
        return iter(self.keys())

    def __setitem__(self, key, value):
        self._update_kv(key, value)

    def __getitem__(self, key):
        return self.unit_dict[key][self.VALUE_INDEX]

    def get(self, key, default=None):
        unit = self.unit_dict.get(key)
        if unit is None:
            return default
        else:
            return unit[self.VALUE_INDEX]

    def keys(self):
        return self.unit_dict.keys()

    def items(self):
        for key, unit in self.unit_dict.items():
            yield key, unit[self.VALUE_INDEX]

    def values(self):
        for unit in self.unit_dict.values():
            yield unit[self.VALUE_INDEX]

    def __contains__(self, key):
        return key in self.unit_dict

    def update(self, *args, **kwargs):
        for key, value in chain(args, kwargs.items()):
            self._update_kv(key, value)

    def __repr__(self):
        return f'{self.__class__.__name__}({self.max_size}, {repr(dict(self.items()))})'


@dataclass
class LRUDictUnit:
    key: 'typing.Any'
    value: 'typing.Any'
    node: DoublyLinkedListNode

    
class LRUDict:
    '''
    Keep limited key-value pairs, and remove Least Recently Used (LRU) key-value pairs.

    Example:

    >>> dic = LRUDict(3)
    >>> dic
    LRUDict(3, {})

    >>> dic['a'] = 1
    >>> dic['b'] = 2
    >>> dic['c'] = 3
    >>> dic
    LRUDict(3, {'a': 1, 'b': 2, 'c': 3})

    >>> dic['a'] = 4
    >>> dic
    LRUDict(3, {'a': 4, 'b': 2, 'c': 3})

    >>> dic['b'], dic['c']  # access without updating values
    (2, 3)
    >>> dic['d'] = 5
    >>> dic
    LRUDict(3, {'b': 2, 'c': 3, 'd': 5})

    >>> dic['e'] = 6
    >>> dic
    LRUDict(3, {'c': 3, 'd': 5, 'e': 6})

    >>> dic['d']
    5

    >>> dic.get('d')
    5

    >>> dic.get('a') is None
    True
    '''

    # Unit = namedlist('Unit', 'key, value, valid')

    KEY_INDEX = 0
    VALUE_INDEX = 1
    NODE_INDEX = 2

    def __init__(self, max_size, /, *args, **kwargs):
        assert max_size > 0
        self.dll = DoublyLinkedList()
        self.unit_dict = {}
        self.max_size = max_size

        for key, value in chain(args, kwargs.items()):
            self._update_kv(key, value)

    def _update_kv(self, key, value):
        if key in self.unit_dict:
            unit = self.unit_dict[key]
            self.dll.send_node_to_rightmost(unit.node)
            unit.value = value
        else:
            if len(self.unit_dict) >= self.max_size:
                lru_unit = self.dll.popleft()
                del self.unit_dict[lru_unit.key]

            unit = LRUDictUnit(key, value, None)
            self.dll.append(unit)
            unit.node = self.dll.tail
            self.unit_dict[key] = unit

    def __iter__(self):
        return iter(self.keys())

    # def __setitem__(self, key, value):
    #     self._update_kv(key, value)

    __setitem__ = _update_kv

    def __getitem__(self, key):
        # return self.unit_dict[key].value
        unit = self.unit_dict[key]
        self.dll.send_node_to_rightmost(unit.node)
        return unit.value

    def get(self, key, default=None):
        unit = self.unit_dict.get(key)
        if unit is None:
            return default
        else:
            return unit.value

    def keys(self):
        return self.unit_dict.keys()

    def items(self):
        for key, unit in self.unit_dict.items():
            yield key, unit.value

    def values(self):
        for unit in self.unit_dict.values():
            yield unit.value

    def __contains__(self, key):
        return key in self.unit_dict

    def update(self, *args, **kwargs):
        for key, value in chain(args, kwargs.items()):
            self._update_kv(key, value)

    def __repr__(self):
        return f'{self.__class__.__name__}({self.max_size}, {repr(dict(self.items()))})'
