
from dhnamlib.pylib.constant import NO_VALUE


# 'indexer' is slower than normal 'dict'
class indexer:
    '''
    Example:

    >>> char_indexer = indexer([[1, 'a'], [3, 'b'], [6, 'c'], [3, 'd']])
    >>> char_indexer
    {1: 'a', 3: 'd', 6: 'c'}
    >>> char_indexer[10] = 'f'
    >>> char_indexer
    {1: 'a', 3: 'd', 6: 'c', 10: 'f'}
    >>> del char_indexer[3]
    >>> char_indexer
    {1: 'a', 6: 'c', 10: 'f'}
    >>> del char_indexer[10]
    >>> char_indexer
    {1: 'a', 6: 'c'}
    '''

    def __init__(self, pairs):
        max_index = max(k for k, v in pairs)
        self._list = [NO_VALUE for _ in range(max_index + 1)]
        for key, value in pairs:
            self._list[key] = value
        self._len = len(set(k for k, v in pairs))

    def items(self):
        for key, value in enumerate(self._list):
            if value is not NO_VALUE:
                yield key, value

    def __iter__(self):
        for key, value in self.items():
            yield key

    def values(self):
        for key, value in self.items():
            yield value

    def __contains__(self, key):
        assert key >= 0
        assert isinstance(key, int)
        return key < len(self._list) and self._list[key] is not NO_VALUE

    def __getitem__(self, key):
        if key in self:
            return self._list[key]
        else:
            raise KeyError(key)

    def get(self, key, default_value=None):
        try:
            return self[key]
        except KeyError:
            return default_value

    def __setitem__(self, key, value):
        if key >= len(self._list):
            self._list.extend(NO_VALUE for _ in range(key - len(self._list) + 1))
        if self._list[key] is NO_VALUE:
            self._len += 1
        self._list[key] = value

    def __delitem__(self, key):
        assert key in self
        self._list[key] = NO_VALUE
        if key + 1 == len(self._list):
            for idx in reversed(range(len(self._list))):
                if self._list[idx] is not NO_VALUE:
                    break
            del self._list[idx + 1:]
        self._len -= 1

    def update(self, pairs):
        for k, v in pairs:
            self[k] = v

    def __len__(self):
        return self._len

    def __repr__(self):
        return repr(dict(self.items()))
