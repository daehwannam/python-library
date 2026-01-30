
class UnusableObject:
    def __init__(self, name=None):
        self._name = name

    def __bool__(self):
        raise Exception(f'{self._get_name_repr} cannot be used as a boolean value.')

    def __eq__(self):
        raise Exception(f'{self._get_name_repr} cannot be used with an equality operator.')

    def _get_name_repr(self):
        if self._name is None:
            return f'The {self.__class__} object'
        else:
            return f'The "{self._name}" object'


class AbstractObject(UnusableObject):
    pass


Abstract = AbstractObject('Abstract')


class NoValueObject(UnusableObject):
    pass


NO_VALUE = NoValueObject('NO_VALUE')


class PlaceholderObject(UnusableObject):
    pass


PLACEHOLDER = PlaceholderObject()


def is_not_no_value(obj):
    return obj is not NO_VALUE
