
def getattr_or_default(obj, attr, default_value=None):
    if hasattr(obj, attr):
        return getattr(obj, attr)
    else:
        return default_value
