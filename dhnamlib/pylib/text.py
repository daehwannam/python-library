
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
