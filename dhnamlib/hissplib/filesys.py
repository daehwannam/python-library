
from ..pylib import filesys as py_filesys


def lissp_save_exprs(lissp_exprs, path):
    py_filesys.write_lines(path, lissp_exprs)


lissp_load_exprs = py_filesys.read_lines
