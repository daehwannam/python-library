
import re


def splititer(pattern, string):
    '''
    >>> string = 'The full name is {first_name} {middle_name} {last_name}'
    >>> pattern = re.compile(r'{([^{}]*)}')

    >>> tuple(splititer(pattern, string))
    ('The full name is ', <re.Match object; span=(17, 29), match='{first_name}'>, ' ', <re.Match object; span=(30, 43), match='{middle_name}'>, ' ', <re.Match object; span=(44, 55), match='{last_name}'>, '')

    >>> splits = []
    >>> for split_obj in splititer(pattern, string):
    ...     if isinstance(split_obj, re.Match):
    ...         match_obj = split_obj
    ...         splits.append(match_obj.group())
    ...     else:
    ...         splits.append(split_obj)

    >>> splits
    ['The full name is ', '{first_name}', ' ', '{middle_name}', ' ', '{last_name}', '']
    '''

    position = 0

    for match_obj in re.finditer(pattern, string):
        span = match_obj.span()
        # content = match_obj.group(group=1)

        yield string[position: span[0]]
        position = span[1]

        yield match_obj

    yield string[position:]


def ismatch(obj):
    return isinstance(obj, re.Match)


def splitenum(pattern, string):
    '''
    >>> string = 'The full name is {first_name} {middle_name} {last_name}'
    >>> pattern = re.compile(r'{([^{}]*)}')

    >>> tuple(splitenum(pattern, string))
    ((None, 'The full name is '), (0, <re.Match object; span=(17, 29), match='{first_name}'>), (None, ' '), (1, <re.Match object; span=(30, 43), match='{middle_name}'>), (None, ' '), (2, <re.Match object; span=(44, 55), match='{last_name}'>), (None, ''))

    >>> splits = []
    >>> for match_index, split_obj in splitenum(pattern, string):
    ...     if match_index is None:
    ...         splits.append(split_obj)
    ...     else:
    ...         match_obj = split_obj
    ...         splits.append(match_obj.group())

    >>> splits
    ['The full name is ', '{first_name}', ' ', '{middle_name}', ' ', '{last_name}', '']
    '''

    position = 0

    for match_index, match_obj in enumerate(re.finditer(pattern, string)):
        span = match_obj.span()
        # content = match_obj.group(group=1)

        yield None, string[position: span[0]]
        position = span[1]

        yield match_index, match_obj

    yield None, string[position:]
