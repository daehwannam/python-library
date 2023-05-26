
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
