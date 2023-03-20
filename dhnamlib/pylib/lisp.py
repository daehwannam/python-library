
from dhnamlib.pylib.iteration import partition

def parse_hy_args(*symbols):
    '''
    e.g. parse_hy_args(*'100 200 300 :x 400 :y 500'.split())
    '''
    args = []

    for idx, symbol in enumerate(symbols):
        if symbol.startswith(':'):
            break
        else:
            args.append(symbol)
    else:
        idx += 1

    def remove_colon(k):
        assert k[0] == ':'
        return k[1:]

    pairs = partition(symbols[idx:], 2)
    kwargs = dict((remove_colon(k), v) for k, v in pairs)

    return args, kwargs
