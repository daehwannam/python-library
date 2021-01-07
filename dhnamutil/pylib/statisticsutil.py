
def add_assign_dict(d1, d2):
    for k, v in d2.items():
        d1[k] = d1.get(k, 0) + v


def div_dict(d, number):
    return type(d)([k, v / number] for k, v in d.items())
