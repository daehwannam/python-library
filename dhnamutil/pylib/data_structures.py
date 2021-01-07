
from functools import reduce
import re
import heapq
from . import min_max_heap
from . import algorithms


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
