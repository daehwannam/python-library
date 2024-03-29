
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
            self._evaluate()

        return self.obj

    def evaluate(self):
        self._evaluate()
        return self.obj

    def _evaluate(self):
        self.obj = self.fn(*self.args, **self.kwargs)
        self.evaluated = True


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


class LazyProxy:
    '''
    Example:

    >>> dic = LazyProxy(lambda: dict(a=10, b=20))
    >>> dic
    LazyProxy({'a': 10, 'b': 20})
    >>> list(dic.items())
    [('a', 10), ('b', 20)]
    >>> dic['a'], dic['b']
    (10, 20)
    >>> dic['c'] = 30
    >>> dic
    LazyProxy({'a': 10, 'b': 20, 'c': 30})
    '''

    def __init__(self, fn, *args, **kwargs):
        super().__setattr__('_lazy_obj', LazyEval(fn, *args, **kwargs))
        # self._lazy_obj = LazyEval(fn, *args, **kwargs)

    def get_instance(self):
        return self._lazy_obj.get()

    def __getattr__(self, name):
        return getattr(self.get_instance(), name)

    def __setattr__(self, name, value):
        setattr(self.get_instance(), name, value)

    def __call__(self, *args, **kwargs):
        return self.get_instance().__call__(*args, **kwargs)

    def __repr__(self):
        # class_name = LazyProxy.__module__ + '.' + LazyProxy.__qualname__
        return '{}({})'.format(self.__class__.__name__, repr(self.get_instance()))

    def __getitem__(self, key):
        return self.get_instance().__getitem__(key)

    def __setitem__(self, key, value):
        self.get_instance().__setitem__(key, value)


class DynamicLazyProxy(LazyProxy):
    '''
    It's same with LazyProxy except `get_instance` always evaluates the LazyEval object.
    '''

    def get_instance(self):
        return self._lazy_obj.evaluate()
