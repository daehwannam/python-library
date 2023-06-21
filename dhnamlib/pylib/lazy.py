
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


def eval_if_lazy(obj):
    if isinstance(obj, LazyEval):
        return obj.get()
    else:
        return obj
