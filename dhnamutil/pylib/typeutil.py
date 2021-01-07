
def isanyinstance(obj, types):
    return is_any_instance(types, obj)


def is_any_instance(types, obj):
    # return any(isinstance(obj, typ) for typ in types)
    return isinstance(obj, tuple(types))
