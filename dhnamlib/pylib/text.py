
import re


def parse_bool(text):
    lower_text = text.lower()
    if lower_text == "true":
        return True
    elif lower_text == "false":
        return False
    else:
        raise Exception("{} is not allowed as a bool value".format(text))


# def parse_bool(s):
#     assert s.lower() in ['true', 'false']
#     return s.lower() == 'true'


def parse_num(s):
    if s == 'inf':
        return float('inf')
    else:
        return int(s)


def one_space(s):
    return ' '.join(s.split())


def get_paren_index_pairs(s):
    # https://stackoverflow.com/a/29992019

    opening_stack = []  # stack of indices of opening parentheses
    pairs = []

    for i, c in enumerate(s):
        if c == '(':
            opening_stack.append(i)
        elif c == ')':
            try:
                pairs.append((opening_stack.pop(), i))
            except IndexError:
                print('Too many closing parentheses')
    if opening_stack:  # check if stack is empty afterwards
        print('Too many opening parentheses')

    return pairs


def split_into_vars(s):
    return s.replace(',', ' ').strip().split()


first_cap_re = re.compile('(.)([A-Z][a-z]+)')
all_cap_re = re.compile('([a-z0-9])([A-Z])')


def camel_to_symbol(name):
    s1 = first_cap_re.sub(r'\1-\2', name)
    return all_cap_re.sub(r'\1-\2', s1).lower()


def camel_to_snake(name):
    s1 = first_cap_re.sub(r'\1-\2', name)
    return all_cap_re.sub(r'\1-\2', s1).lower()


var_regex = re.compile(r'{[_a-zA-Z][_a-zA-Z0-9]*}')


def flexible_format(string, *args, **kwargs):
    '''
    Example:
    >>> flexible_format('f(x) = {x + {offset}}', offset=100)
    'f(x) = {x + 100}'
    '''
    def replace_curly(s):
        return s.replace('{', '{{').replace('}', '}}')

    splits = []
    index = 0
    for match_obj in var_regex.finditer(string):
        span = match_obj.span()
        splits.append(replace_curly(string[index: span[0]]))
        splits.append(match_obj.group())
        index = span[1]

    splits.append(replace_curly(string[index:]))

    template = ''.join(splits)

    return template.format(*args, **kwargs)


key_regex = re.compile(r'{([^\s{}]+)}')

def replace_keys(string, pairs, key_regex=key_regex):
    '''
    Replace keys.
    >>> replace_keys('f({name-1}, {name-2}) = {{name-1} + {name-2}}',
    ...              [['name-1', 'x'], ['name-2', 'y']])
    'f(x, y) = {x + y}'
    Example:

    '''
    kv_dict = pairs if isinstance(pairs, dict) else dict(pairs)
    splits = []
    index = 0
    for match_obj in key_regex.finditer(string):
        span = match_obj.span()
        splits.append(string[index: span[0]])
        splits.append(kv_dict[match_obj.group(1)])
        index = span[1]

    splits.append(string[index:])

    return ''.join(splits)
