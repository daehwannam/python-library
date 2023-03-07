
from functools import reduce
import re
import heapq
from collections import deque, defaultdict

from . import min_max_heap
from . import algorithms
from .struct import namedlist


class HeapPQ:
    def __init__(self):
        self.heap = []
        self.item_num = 0

    def push(self, priority, item):
        self.item_num += 1
        heapq.heappush(self.heap, (priority, self.item_num, item))

    def pop(self):  # return an item of the smallest value of priority
        priority, item_num, item = heapq.heappop(self.heap)
        return item

    def pushpop(self, priority, item):
        self.item_num += 1
        priority, item_num, item = heapq.heappushpop(self.heap, (priority, self.item_num, item))
        return item

    @property
    def root(self):
        priority, item_num, item = self.heap[0]
        return item

    def prune(self):
        pass

    def __bool__(self):
        return bool(self.heap)

    def __iter__(self):
        for priority, item_num, item in self.heap:
            yield item

    def __len__(self):
        return len(self.heap)


class LimitedPQ:
    def __init__(self, size):
        self.max_size = size
        self.lst = []
        self.item_num = 0

    def push(self, priority, item):
        self.item_num += 1
        self.lst.append((priority, self.item_num, item))

    def pop(self):
        min_idx = min(range(len(self.lst)), key=lambda idx: self.lst[idx])
        priority, item_num, item = self.lst[min_idx]
        self.lst[min_idx] = self.lst[len(self.lst) - 1]
        self.lst.pop()
        return item

    def prune(self):
        if len(self.lst) > self.max_size:
            # self.lst.sort()
            algorithms.quickselect(self.lst, self.max_size)
            del self.lst[-(len(self.lst) - self.max_size):]

    def __bool__(self):
        return bool(self.lst) > 0


# # with min-max-heap
# class LimitedPQ:
#     def __init__(self, size):
#         self.max_size = size
#         self.heap = min_max_heap.MinMaxHeap(size)
#         self.item_num = 0

#     def push(self, priority, item):
#         self.item_num += 1
#         self.heap.push((priority, self.item_num, item))
#         if len(self.heap) > self.max_size:
#             self.heap.pop_max()

#     def pop(self):
#         priority, item_num, item = self.heap.pop_min()
#         return item

#     def prune(self):
#         pass

#     def __bool__(self):
#         return len(self.heap) > 0


class LRUSet:
    Unit = namedlist('Unit', 'item, valid')

    def __init__(self, max_size):
        self.q = deque()
        self.unit_dict = {}
        self.max_size = max_size
        self.size = 0

    def add(self, item):
        if item in self.unit_dict:
            self.unit_dict[item].valid = False
        elif self.size < self.max_size:
            self.size += 1
        else:
            lr_unit = self.q.popleft()
            while not lr_unit.valid:
                lr_unit = self.q.popleft()
            del self.unit_dict[lr_unit.item]

        unit = self.Unit(item, True)
        self.q.append(unit)
        self.unit_dict[item] = unit

        assert len(self.unit_dict) <= self.max_size

    def __iter__(self):
        value_idx = self.Unit.get_attr_idx('item')
        for unit in self.unit_dict.values():
            yield unit[value_idx]

    def __repr__(self):
        return f'{self.__class__.__name__}(max_size={self.max_size}, {repr(set(self))})'


class LRUDict:
    Unit = namedlist('Unit', 'key, value, valid')

    def __init__(self, max_size):
        self.q = deque()
        self.unit_dict = {}
        self.max_size = max_size
        self.size = 0

    def update_kv(self, key, value):
        if key in self.unit_dict:
            self.unit_dict[key].valid = False
        elif self.size < self.max_size:
            self.size += 1
        else:
            lr_unit = self.q.popleft()
            while not lr_unit.valid:
                lr_unit = self.q.popleft()
            del self.unit_dict[lr_unit.key]

        unit = self.Unit(key, value, True)
        self.q.append(unit)
        self.unit_dict[key] = unit

        assert len(self.unit_dict) <= self.max_size

    def __iter__(self):
        return iter(self.keys())

    def __setitem__(self, key, value):
        self.update_kv(key, value)

    def __getitem__(self, key):
        return self.unit_dict[key].value

    def keys(self):
        return self.unit_dict.keys()

    def items(self):
        value_idx = self.Unit.get_attr_idx('value')
        for key, unit in self.unit_dict.items():
            yield key, unit[value_idx]

    def values(self):
        value_idx = self.Unit.get_attr_idx('value')
        for unit in self.unit_dict.values():
            yield unit[value_idx]

    def __repr__(self):
        return f'{self.__class__.__name__}(max_size={self.max_size}, {repr(dict(self.items()))})'


class LIFOSet:
    def __init__(self):
        self.count_dict = {}
        self.count = 0

    def add(self, item):
        self.count_dict[item] = self.count
        self.count += 1

    def __iter__(self):
        for k, v in sorted(self.count_dict.items(), key=lambda x: - x[1]):
            yield k

    def __repr__(self):
        return f'{self.__class__.__name__}({repr(set(self))})'


class LIFODict:
    def __init__(self):
        self.count_dict = {}
        self.value_dict = {}
        self.count = 0

    def update_kv(self, key, value):
        self.count_dict[key] = self.count
        self.count += 1
        self.value_dict[key] = value

    def __setitem__(self, key, value):
        self.update_kv(key, value)

    def __getitem__(self, key):
        return self.value_dict[key]

    def __iter__(self):
        return iter(self.keys())

    def keys(self):
        for k, v in sorted(self.count_dict.items(), key=lambda x: - x[1]):
            yield k

    def items(self):
        for k in self:
            yield k, self.value_dict[k]

    def values(self):
        for k in self:
            yield self.value_dict[k]

    def __repr__(self):
        return f'{self.__class__.__name__}({repr(dict(self))})'


class PriorityDict:
    Unit = namedlist('Unit', 'key, value, valid')

    def __init__(self):
        self.pq = HeapPQ()
        self.unit_dict = {}

    def update(self, priority, key, value):
        unit = PriorityDict.Unit(key, value, True)
        self.pq.push(priority, unit)

        existing_unit = self.unit_dict.get(key)
        if existing_unit is not None:
            unit.valid = False
        self.unit_dict[key] = unit

    def pop(self):
        unit = self.pq.pop()
        while not unit.valid:
            unit = self.pq.pop()
        del self.self.unit_dict[unit.key]
        return unit.value

    def __iter__(self):
        return self.keys()

    def keys(self):
        return self.unit_dict.keys()

    def items(self):
        value_idx = self.Unit.get_attr_idx('value')
        for key, unit in self.unit_dict.items():
            yield key, unit[value_idx]

    def values(self):
        value_idx = self.Unit.get_attr_idx('value')
        for unit in self.unit_dict.values():
            yield unit[value_idx]


class BipartiteGraph:
    def __init__(self, inverse_init=True):
        self.graph = defaultdict(set)
        self.inverse = BipartiteGraph(inverse=False)
        self.inverse.inverse = self

    def add_edge(self, source, target):
        assert source not in self.inverse.graph
        assert target not in self.graph

        self.graph[source].add(target)
        self.inverse.graph[target].add(source)

    def get_vertices(self, source):
        return self.graph[source]
