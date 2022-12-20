
def parse_bool(text):
    lower_text = text.lower()
    if lower_text == "true":
        return True
    elif lower_text == "false":
        return False
    else:
        raise Exception("{} is not allowed as a bool value".format(text))


def parse_num(s):
    if s == 'inf':
        return float('inf')
    else:
        return int(s)
