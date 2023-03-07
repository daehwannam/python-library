

def parse_bool(s):
    assert s.lower() in ['true', 'false']
    return s.lower() == 'true'


def parse_int(s):
    if s == 'inf':
        return float('inf')
    else:
        return int(s)
