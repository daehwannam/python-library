
from dataclasses import dataclass
import typing
from .function import identity
from .constant import NO_VALUE


_dll_empty_error_message = 'The doubly linked list is empty.'


class DoublyLinkedList:
    """
    >>> dll = DoublyLinkedList()
    >>> dll.append('a')
    >>> dll.appendleft('b')
    >>> dll.append('c')
    >>> dll.appendleft('d')
    >>> dll
    DoublyLinkedList(['d', 'b', 'a', 'c'])
    >>> len(dll)
    4

    >>> dll.popleft()
    'd'
    >>> dll.pop()
    'c'
    >>> dll.popleft()
    'b'
    >>> dll.pop()
    'a'
    >>> dll
    DoublyLinkedList([])
    >>> len(dll)
    0

    >>> dll = DoublyLinkedList()
    >>> dll.extend(range(2))
    >>> dll
    DoublyLinkedList([0, 1])
    >>> dll.extend(range(2, 4))
    >>> dll
    DoublyLinkedList([0, 1, 2, 3])
    >>> len(dll)
    4

    >>> dll.pop()
    3
    >>> dll.pop()
    2
    >>> dll.pop()
    1
    >>> dll.pop()
    0
    >>> dll
    DoublyLinkedList([])
    >>> len(dll)
    0

    >>> dll.extendleft(range(2))
    >>> dll
    DoublyLinkedList([1, 0])
    >>> dll.extendleft(range(2, 4))
    >>> dll
    DoublyLinkedList([3, 2, 1, 0])
    >>> len(dll)
    4

    >>> dll.popleft()
    3
    >>> dll.popleft()
    2
    >>> dll.popleft()
    1
    >>> dll.popleft()
    0
    >>> dll
    DoublyLinkedList([])
    >>> len(dll)
    0

    >>> dll.extend(['A', 'B', 'C'])
    >>> node = dll.findnode('b', key=lambda x: x.lower())
    >>> node.item
    'B'
    >>> dll.remove_node(node)
    >>> dll
    DoublyLinkedList(['A', 'C'])
    >>> len(dll)
    2
    >>> dll.remove_node(dll.head)
    >>> dll
    DoublyLinkedList(['C'])
    >>> dll.remove_node(dll.tail)
    >>> dll
    DoublyLinkedList([])
    >>> len(dll)
    0

    >>> dll.extend(['A', 'B', 'C', 'D', 'E'])
    >>> node = dll.findnode('c', key=lambda x: x.lower())
    >>> dll.send_node_to_rightmost(node)
    >>> dll
    DoublyLinkedList(['A', 'B', 'D', 'E', 'C'])
    >>> dll.send_node_to_leftmost(node)
    >>> dll
    DoublyLinkedList(['C', 'A', 'B', 'D', 'E'])

    >>> dll = DoublyLinkedList(['X'])
    >>> node = dll.head
    >>> dll.send_node_to_rightmost(node)
    >>> dll
    DoublyLinkedList(['X'])
    >>> dll.send_node_to_leftmost(node)
    >>> dll
    DoublyLinkedList(['X'])
    """

    def __init__(self, items=[]):
        self.head = None
        self.tail = None
        self._length = 0
        self.extend(items)

    def append(self, item):
        node = DoublyLinkedListNode(item, self.tail, None)
        if self.tail is None:
            self.head = node
        else:
            self.tail.next = node
        self.tail = node
        self._length += 1

    def appendleft(self, item):
        node = DoublyLinkedListNode(item, None, self.head)
        if self.head is None:
            self.tail = node
        else:
            self.head.prev = node
        self.head = node
        self._length += 1

    def extend(self, items=[]):
        iterator = iter(items)
        try:
            first_item = next(iterator)
            node = DoublyLinkedListNode(first_item, self.tail, None)
            if self.tail is None:
                self.head = node
            else:
                self.tail.next = node
            self._length += 1

            try:
                while True:
                    item = next(iterator)
                    new_node = DoublyLinkedListNode(item, node, None)
                    node.next = new_node
                    node = new_node
                    self._length += 1
            except StopIteration:
                self.tail = node
        except StopIteration:
            pass

    def extendleft(self, items=[]):
        iterator = iter(items)
        try:
            first_item = next(iterator)
            node = DoublyLinkedListNode(first_item, None, self.head)
            if self.head is None:
                self.tail = node
            else:
                self.head.prev = node
            self._length += 1

            try:
                while True:
                    item = next(iterator)
                    new_node = DoublyLinkedListNode(item, None, node)
                    node.prev = new_node
                    node = new_node
                    self._length += 1
            except StopIteration:
                self.head = node
        except StopIteration:
            pass

    def pop(self):
        return self.pop_node().item

    def popleft(self):
        return self.popleft_node().item

    def pop_node(self):
        node = self.tail
        try:
            self.tail = node.prev
            if self.tail is None:
                self.head = None
            else:
                self.tail.next = None
            self._length -= 1
        except AttributeError:
            raise Exception(_dll_empty_error_message) from None
            assert node is None
        return node

    def popleft_node(self):
        node = self.head
        try:
            self.head = node.next
            if self.head is None:
                self.tail = None
            else:
                self.head.prev = None
            self._length -= 1
        except AttributeError:
            raise Exception(_dll_empty_error_message) from None
            assert node is None
        return node

    def remove_node(self, node):
        prev = node.prev
        next = node.next

        if prev is None:
            self.head = next
            if next is None:
                self.tail = None
            else:
                next.prev = None
        else:
            prev.next = next
            if next is None:
                self.tail = prev
            else:
                next.prev = prev

        self._length -= 1

    def send_node_to_rightmost(self, node):
        self.remove_node(node)

        node.prev = self.tail
        node.next = None

        if self.tail is None:
            self.head = node
        else:
            self.tail.next = node
        self.tail = node
        self._length += 1

    def send_node_to_leftmost(self, node):
        self.remove_node(node)

        node.prev = None
        node.next = self.head

        if self.head is None:
            self.tail = node
        else:
            self.head.prev = node
        self.head = node
        self._length += 1

    def __bool__(self):
        return self.head is not None

    def __iter__(self):
        node = self.head
        while node is not None:
            yield node.item
            node = node.next

    def __reversed__(self):
        node = self.tail
        while node is not None:
            yield node.item
            node = node.prev

    def iternodes(self):
        node = self.head
        while node is not None:
            yield node
            node = node.next

    def reversenodes(self):
        node = self.tail
        while node is not None:
            yield node
            node = node.prev

    def __repr__(self):
        return f'{self.__class__.__name__}({repr(list(self))})'

    def __len__(self):
        return self._length

    def findnode(self, target, key=identity, from_left=True, default=NO_VALUE):
        iterator = iter(self.iternodes() if from_left else self.reversenodes())
        while True:
            try:
                node = next(iterator)
                if key(node.item) == target:
                    return node
            except StopIteration:
                if default is NO_VALUE:
                    raise ValueError('The target {repr(target)} cannot be found.') from None
                else:
                    return default


@dataclass
class DoublyLinkedListNode:
    item: 'typing.Any'
    prev: 'DoublyLinkedListNode'
    next: 'DoublyLinkedListNode'

    def simple_repr(self):
        return f'{self.__class__.__name__}(item={repr(self.item)})'
