
import importlib
import os


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


def is_package_init_file(file_path):
    return os.path.basename(file_path) == '__init__.py'


def is_package(package_or_module):
    return package_or_module.__file__ is None or \
        is_package_init_file(package_or_module.__file__)


class ModuleAccessor:
    '''
    Example:

    >>> module_accessor = ModuleAccessor('dhnamlib')
    >>> module_accessor.pylib.filesys.__name__
    'dhnamlib.pylib.filesys'
    '''

    def __init__(self, package_name, package=None):
        self.package_name = package_name
        if package is None:
            package = importlib.import_module(package_name)
            assert is_package(package), f'{package_name} is a module rather than a package'
        self.package = package
        self.cache = dict()

    def __getattr__(self, name):
        extended_name = self.package_name + '.' + name
        if extended_name not in self.cache:
            package_or_module = importlib.import_module(extended_name)
            self.cache[extended_name] = \
                ModuleAccessor(extended_name, package_or_module) if is_package(package_or_module) else \
                package_or_module
        return self.cache[extended_name]
