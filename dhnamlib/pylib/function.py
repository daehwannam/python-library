
# it can be replaced with cache decorator

# def make_get_singleton(name, func):
#     def get_singleton():
#         if name not in singleton_dict:
#             singleton_dict[name] = func()
#         return singleton_dict[name]
#     return get_singleton


# singleton_dict = {}

def identity(x):
    return x


def maps(fn, coll):
    "map only for side effects"
    for elem in coll:
        fn(elem)


def starmaps(fn, coll):
    for elem in coll:
        fn(*elem)


def starmap(fn, coll):
    for elem in coll:
        yield fn(*elem)
