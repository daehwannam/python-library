
from itertools import chain
import reprlib

# from dhnamlib.pylib.heap import PriorityCache
# from dhnamlib.pylib.klass import subclass, forbid

import pygtrie
from pygtrie import _NoChildren, _OneChild, _Children


# This code is tested with pygtrie==2.5.0 .


class SequenceTrie:
    """
    >>> trie = SequenceTrie(
    ...     [[0],
    ...      [1],
    ...      [2],
    ...      [0, 1, 2],
    ...      [0, 1, 3],
    ...      [0, 1, 4],
    ...      [0, 1, 2, 3, 4],
    ...      [0, 1, 2, 3, 5, 6],
    ...      [0, 1, 2, 3, 5, 7]])
    >>> trie
    {(2,), (0, 1, 4), (0, 1, 2), (0, 1, 3), (0, 1, 2, 3, 5, 7), (1,), (0, 1, 2, 3, 5, 6), (0,), (0, 1, 2, 3, 4)}

    >>> list(trie.endable_elems([0, 1]))
    [2, 3, 4]
    >>> list(trie.continuable_elems([0, 1]))
    [2]

    >>> list(trie.endable_elems([0, 1, 2]))
    []
    >>> list(trie.continuable_elems([0, 1, 2]))
    [3]

    >>> list(trie.endable_elems([0, 1, 2, 3, 5]))
    [6, 7]
    >>> list(trie.continuable_elems([0, 1, 2, 3, 5]))
    []

    >>> trie.has_endable_elem([0, 1])
    True
    >>> trie.has_continuable_elem([0, 1])
    True

    >>> trie.has_endable_elem([0, 1, 2])
    False
    >>> trie.has_continuable_elem([0, 1, 2, 3, 5])
    False

    >>> trie.is_endable_elem([0, 1], 2)
    True
    >>> trie.is_continuable_elem([0, 1], 2)
    True

    >>> trie.is_endable_elem([0, 1, 2], 3)
    False
    >>> trie.is_continuable_elem([0, 1, 2, 3, 5], 6)
    False

    """

    def __init__(self, seqs):
        self._len = 0
        self._trie = pygtrie.Trie()
        for seq in seqs:
            self.add(seq)

    def add(self, seq):
        assert len(seq) > 0

        except_last, last = seq[:-1], seq[-1]
        node = self._trie._set_node(except_last, set(), only_if_missing=True)
        node.value.add(last)

        self._len += 1

    def remove(self, seq):
        assert len(seq) > 0

        except_last, last = seq[:-1], seq[-1]
        node, trace = self._get_prefix_node(except_last)
        node.value.remove(last)
        if len(node.value) == 0:
            self._trie._pop_value(trace)

        self._len -= 1

    def update(self, seqs):
        for seq in seqs:
            self.add(seq)

    def _get_prefix_node(self, prefix):
        try:
            prefix_node, trace = self._trie._get_node(prefix)
        except KeyError:
            raise KeyError(f'Unknown prefix {repr(prefix)}') from None
        return prefix_node, trace

    def _get_endable_elem_set(self, prefix_node):
        # prefix_node, trace = self._get_prefix_node(prefix)
        if isinstance(prefix_node.value, set):
            return prefix_node.value
        else:
            return set()

    def _get_continuable_elem_dict(self, prefix_node):
        # prefix_node, trace = self._get_prefix_node(prefix)
        children = prefix_node.children
        if isinstance(children, _NoChildren):
            return {}
        elif isinstance(children, _OneChild):
            return {children.step: children.node}
        else:
            assert isinstance(children, _Children)
            return children

    def endable_elems(self, prefix):
        prefix_node, trace = self._get_prefix_node(prefix)
        return iter(self._get_endable_elem_set(prefix_node))

    def continuable_elems(self, prefix):
        prefix_node, trace = self._get_prefix_node(prefix)
        return iter(self._get_continuable_elem_dict(prefix_node))

    # def _endable_elems_from_node(self, node):
    #     if isinstance(node.value, set):
    #         return iter(node.value)
    #     else:
    #         return iter(())

    # def _continuable_elems_from_node(self, node):
    #     for elem, node in node.children.iteritems():
    #         yield elem

    def has_endable_elem(self, prefix):
        prefix_node, trace = self._get_prefix_node(prefix)
        endable_elem_set = self._get_endable_elem_set(prefix_node)
        return len(endable_elem_set) > 0

    def has_continuable_elem(self, prefix):
        prefix_node, trace = self._get_prefix_node(prefix)
        continuable_elem_dict = self._get_continuable_elem_dict(prefix_node)
        return len(continuable_elem_dict) > 0

    def is_endable_elem(self, prefix, last_elem):
        prefix_node, trace = self._get_prefix_node(prefix)
        endable_elem_set = self._get_endable_elem_set(prefix_node)
        return last_elem in endable_elem_set

    def is_continuable_elem(self, prefix, last_elem):
        prefix_node, trace = self._get_prefix_node(prefix)
        continuable_elem_dict = self._get_continuable_elem_dict(prefix_node)
        return last_elem in continuable_elem_dict

    def elems(self, prefix):
        return chain(self.endable_elems(), self.continuable_elems())

    def __repr__(self):
        return reprlib.repr(set(self))

    def __len__(self):
        return self._len

    def __contains__(self, seq):
        assert len(seq) > 0

        try:
            except_last, last = seq[:-1], seq[-1]
            node, trace = self._get_prefix_node(except_last)
        except KeyError:
            return False
        else:
            if isinstance(node.value, set):
                return last in node.value
            else:
                return False

    def __iter__(self):
        for except_last, endable_elems in self._trie.items():
            for endable_elem in endable_elems:
                yield tuple(chain(except_last, [endable_elem]))

    def __eq__(self, other):
        return tuple(self) == tuple(other)

    def __ne__(self, other):
        return self != other

    def _compute_content(self):
        def get_content(node):
            endable_elems = tuple(sorted(self._get_endable_elem_set(node)))
            sub_content = tuple(sorted(
                (continuable_elem, get_content(child_node))
                for continuable_elem, child_node in node.children.iteritems()))

            return (endable_elems, sub_content)

        return get_content(self._trie._root)
