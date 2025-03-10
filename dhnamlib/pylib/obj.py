
from .constant import NO_VALUE


def getattr_or_default(obj, attr, default_value=None):
    if hasattr(obj, attr):
        return getattr(obj, attr)
    else:
        return default_value


def get_nested_attr(obj, *attrs, default_value=None):
    if len(attrs) == 1 and '.' in attrs[0]:
        attrs = attrs[0].split('.')
    for attr in attrs:
        if hasattr(obj, attr):
            obj = getattr(obj, attr)
        else:
            return default_value
    return obj


def get_nested_item(obj, *keys, default_value=None):
    for key in keys:
        if key in obj:
            obj = obj[key]
        else:
            return default_value
    return obj


def getitem(obj, key, default_value=NO_VALUE):
    '''
    Example:

    >>> getitem('abcde', 1, 'X')
    'b'
    >>> getitem('abcde', 10, 'X')
    'X'
    >>> getitem(dict(a=1, b=2, c=3), 'b', -1)
    2
    >>> getitem(dict(a=1, b=2, c=3), 'x', -1)
    -1
    '''
    try:
        return obj.__getitem__(key)
    except (IndexError, KeyError) as e:
        if default_value is NO_VALUE:
            raise e
        else:
            return default_value


class ObjectCache:
    '''
    Example:

    >>> object_cache = ObjectCache()
    >>> def foo():
    ...     print('`foo` is called')
    ...     return 'something'

    >>> object_cache.set_initializer(foo).__name__
    'foo'
    >>> print('Before getting the object')
    Before getting the object
    >>> object_cache.get_object()
    `foo` is called
    'something'

    >>> object_cache = ObjectCache()
    >>> @object_cache.set_initializer
    ... def bar():
    ...     print('`bar` is called')
    ...     return 'another'

    >>> object_cache.get_object()
    `bar` is called
    'another'
    '''

    def __init__(self):
        self._obj = NO_VALUE
        self._initializer = NO_VALUE
        self._evaluated = False

    def _assert_object_is_unset(self):
        max_repr_len = 30
        assert self._obj is NO_VALUE, f'An object is already set as {repr(self._obj)[:max_repr_len]}'
        assert self._initializer is NO_VALUE, f'An initializer is already set as {repr(self._initializer)[:max_repr_len]}'

    def is_cached(self):
        return (self._obj is not NO_VALUE) or \
            (self._initializer is not NO_VALUE)

    def set_object(self, obj):
        self._assert_object_is_unset()
        self._obj = obj
        return obj

    def set_initializer(self, obj_fn):
        self._assert_object_is_unset()
        self._initializer = obj_fn
        return obj_fn

    def get_object(self):
        if self._obj is not NO_VALUE:
            return self._obj
        elif self._initializer is not NO_VALUE:
            assert self._evaluated is False
            self._obj = self._initializer()
            self._evaluated = True
            return self._obj
        else:
            raise Exception('Either object or object function does not exist')
