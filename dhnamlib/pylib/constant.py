
class AbstractType:
    pass


Abstract = AbstractType()


class NoValueType:
    pass


NO_VALUE = NoValueType()


def is_not_no_value(obj):
    return obj is not NO_VALUE
