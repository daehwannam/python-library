import os
import importlib

from hissp.reader import transpile


def import_lissp(module_name, force_compile=False):
    module_path = os.sep.join(module_name.split('.'))
    if 'PYTHONPATH' in os.environ:
        module_path = os.path.join(os.environ['PYTHONPATH'], module_path)
    lissp_time = os.path.getmtime(module_path + '.lissp')
    if os.path.isfile(module_path + '.py'):
        py_time = os.path.getmtime(module_path + '.py')
    else:
        py_time = float('-inf')

    if force_compile or lissp_time > py_time:
        # transpile(__package__, module_name)
        transpile(None, module_path)

    return importlib.import_module(module_name)
