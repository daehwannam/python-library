
from .decoration import deprecated
from .function import identity


class LazyEval:
    def __init__(self, fn, *args, **kwargs):
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.evaluated = False
        self.obj = None

    def get(self):
        if not self.evaluated:
            self.obj = self.fn(*self.args, **self.kwargs)
            self.evaluated = True

        return self.obj


def eval_lazy_obj(obj):
    if isinstance(obj, LazyEval):
        return obj.get()
    else:
        return obj


@deprecated
def eval_obj_unless_lazy(obj, lazy):
    if lazy:
        return obj
    else:
        return eval_lazy_obj(obj)


def get_eval_obj_unless_lazy(lazy):
    if lazy:
        return identity
    else:
        return eval_lazy_obj
