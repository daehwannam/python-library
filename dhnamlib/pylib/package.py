
import importlib


def get_ancestor(full_name, gap):
    "'full-name' is the full name of module or package"
    assert gap > 0

    return ".".join(full_name.split(".")[0: -gap])


def get_parent(package):  # parent-package
    "a.b.c --> a.b"

    return get_ancestor(package, 1)


def join(*args):  # join names
    return ".".join(args)


def import_from_module(module_name, obj_names):
    if isinstance(obj_names, str):
        obj_names = [obj_names]
        sole_obj = True
    else:
        sole_obj = False

    module = importlib.import_module(module_name)
    objs = tuple(getattr(module, obj_name) for obj_name in obj_names)

    if sole_obj:
        return objs[0]
    else:
        return objs
