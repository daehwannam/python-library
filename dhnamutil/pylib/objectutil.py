
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
