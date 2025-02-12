import os
import sys
import importlib

from hissp.reader import transpile


def import_lissp(module_name, force_compile=False):
    partial_module_path_without_extension = os.sep.join(module_name.split('.'))
    for python_path in sys.path:
        full_module_path_without_extension = os.path.join(python_path, partial_module_path_without_extension)
        lissp_module_path = full_module_path_without_extension + '.lissp'
        if os.path.isfile(lissp_module_path):
            break
    else:
        raise ModuleNotFoundError(f"No module named '{module_name}'")
    lissp_time = os.path.getmtime(lissp_module_path)
    generated_python_code_path = full_module_path_without_extension + '.py'
    if os.path.isfile(generated_python_code_path):
        py_time = os.path.getmtime(full_module_path_without_extension + '.py')
    else:
        py_time = float('-inf')

    if force_compile or lissp_time > py_time:
        # transpile(__package__, module_name)
        transpile(None, full_module_path_without_extension)

    return importlib.import_module(module_name)
