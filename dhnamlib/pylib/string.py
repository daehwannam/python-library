
def one_space(s):
    return ' '.join(s.split())


def parentehesis_index_pairs(s):
    # https://stackoverflow.com/a/29992019

    opening_stack = []  # stack of indices of opening parentheses
    pairs = []

    for i, c in enumerate(s):
        if c == '(':
            opening_stack.append(i)
        if c == ')':
            try:
                pairs.append((opening_stack.pop(), i))
            except IndexError:
                print('Too many closing parentheses')
    if opening_stack:  # check if stack is empty afterwards
        print('Too many opening parentheses')

    return pairs

def split_by_indices(s, indices):
    last_idx = 0
    for idx in indices:
        yield s[last_idx: idx]
        last_idx = idx
