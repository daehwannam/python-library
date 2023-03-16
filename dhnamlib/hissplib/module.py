import os
import importlib

from hissp.reader import transpile


def import_lissp(module_path, force_compile=False):
    lissp_time = os.path.getmtime(module_path + '.lissp')
    py_time = os.path.getmtime(module_path + '.py')

    if force_compile or lissp_time > py_time:
        transpile(__package__, module_path)

    return importlib.import_module(module_path)
