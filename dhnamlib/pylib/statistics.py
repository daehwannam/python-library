
import random


def add_assign_dict(d1, d2):
    for k, v in d2.items():
        d1[k] = d1.get(k, 0) + v


def div_dict(d, number):
    return type(d)([k, v / number] for k, v in d.items())


def shuffled(items, seed=None):
    if seed is not None:
        _random = random.Random(seed)
    else:
        _random = random
    new_items = list(items)
    _random.shuffle(new_items)
    return new_items


class Number:
    """
    >>> num = Number(10)
    >>> num
    Number(10)
    >>> num + 1
    Number(11)
    >>> num - 1
    Number(9)
    >>> int(num)
    10
    """

    def __init__(self, raw):
        self._raw = raw

    @property
    def raw(self):
        return self._raw

    @staticmethod
    def _convert_to_raw(number):
        if isinstance(number, Number):
            return number._raw
        elif isinstance(number, (int, float)):
            return number
        else:
            raise Exception('Unexpected type.')

    def __eq__(self, other):
        return self._raw == Number._convert_to_raw(other)

    def __ne__(self, other):
        return self._raw != Number._convert_to_raw(other)

    def __lt__(self, other):
        return self._raw < Number._convert_to_raw(other)

    def __le__(self, other):
        return self._raw <= Number._convert_to_raw(other)

    def __gt__(self, other):
        return self._raw > Number._convert_to_raw(other)

    def __ge__(self, other):
        return self._raw >= Number._convert_to_raw(other)

    def __int__(self):
        if isinstance(self._raw, int):
            return self._raw
        else:
            return int(self._raw)

    def __float__(self):
        if isinstance(self._raw, float):
            return self._raw
        else:
            return float(self._raw)

    def __bool__(self):
        return bool(self._raw)

    def __add__(self, other):
        return Number(self._raw + Number._convert_to_raw(other))

    def __sub__(self, other):
        return Number(self._raw - Number._convert_to_raw(other))

    def __repr__(self):
        return f'{self.__class__.__name__}({self._raw})'

    def increase(self, number):
        self._raw += self._convert_to_raw(number)

    def decrease(self, number):
        self._raw -= self._convert_to_raw(number)
