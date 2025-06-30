
class AbstractObject:
    pass


Abstract = AbstractObject()


class NoValueObject:
    pass


NO_VALUE = NoValueObject()


class PlaceholderObject:
    pass


PLACEHOLDER = PlaceholderObject()


def is_not_no_value(obj):
    return obj is not NO_VALUE
