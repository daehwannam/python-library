
def isanyinstance(obj, types):
    return is_any_instance(types, obj)


def is_any_instance(types, obj):
    # return any(isinstance(obj, typ) for typ in types)
    return isinstance(obj, tuple(types))


def creatable(cls, *args, exception_cls=ValueError, **kwargs):
    if isinstance(exception_cls, tuple):
        assert all(issubclass(cls, Exception) for cls in exception_cls)
    else:
        assert issubclass(exception_cls, Exception)

    try:
        cls(*args, **kwargs)
        return True
    except exception_cls:
        return False


def typecast(obj, typ):
    if isinstance(obj, typ):
        return obj
    else:
        return typ(obj)


def is_type(typ):
    '''
    Example:

    >>> is_type(int)
    True
    >>> is_type((int, float))
    True
    >>> is_type(None)
    False
    >>> is_type((int, None))
    False
    '''
    return isinstance(typ, type) or \
        (isinstance(typ, tuple) and
         all(isinstance(obj, type) for obj in typ))
