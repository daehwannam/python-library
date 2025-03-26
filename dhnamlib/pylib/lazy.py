
from .decoration import deprecated, curry
from .function import identity
from .constant import NO_VALUE


class LazyEval:
    def __init__(self, fn, *args, **kwargs):
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.evaluated = False
        self.obj = NO_VALUE

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


# Register
class Register:
    '''
    Example
    >>> register = Register(strategy='lazy')
    >>> name_fn = register.retrieve('name-fn')
    >>>
    >>> @register('name-fn')
    ... def full_name(first, last):
    ...     return ' '.join([first, last])
    >>>
    >>> print(name_fn('John', 'Smith'))
    John Smith
    '''

    STRATEGIES = ('instant', 'lazy', 'conditional')

    def __init__(self, strategy='instant'):
        assert strategy in self.STRATEGIES
        self.strategy = strategy
        self.memory = dict()

    def _normalize_identifier(self, identifier):
        if isinstance(identifier, list):
            identifier = tuple(identifier)
        return identifier

    @curry
    def __call__(self, identifier, obj):
        identifier = self._normalize_identifier(identifier)

        assert identifier not in self.memory, f'"{identifier}" is already registered.'
        self.memory[identifier] = obj
        return obj

    # def __call__(self, identifier, func=None):
    #     if func is None:
    #         def decorator(func):
    #             assert identifier not in self.memory
    #             self.memory[identifier] = func
    #             return func
    #         return decorator
    #     else:
    #         assert identifier not in self.memory
    #         self.memory[identifier] = func

    def retrieve(self, identifier, strategy=None):
        identifier = self._normalize_identifier(identifier)

        if strategy is None:
            strategy = self.strategy
        else:
            assert strategy in self.STRATEGIES

        if (strategy == 'lazy') or (strategy == 'conditional' and identifier not in self.memory):
            registered = LazyRegisterValue(self, identifier)
        else:
            assert strategy in ['instant', 'conditional']
            registered = self.memory[identifier]

        return registered

    @staticmethod
    def _msg_not_registered(identifier):
        return f'"{identifier}" is not registered.'

    def update(self, pairs, **kwargs):
        _pairs = pairs.memory.items() if isinstance(pairs, Register) else pairs
        self.memory.update(_pairs, **kwargs)

    def items(self):
        return self.memory.items()


class MethodRegister(Register):
    '''
    Example:

    >>> class User:
    ...     method_register = MethodRegister()
    ...
    ...     def __init__(self, first_name, last_name):
    ...         self.first_name = first_name
    ...         self.last_name = last_name
    ...         self.register = self.method_register.instantiate(self)
    ...
    ...     @method_register('full_name')
    ...     def get_full_name(self):
    ...         return f'{self.first_name} {self.last_name}'
    ...
    ...     @method_register('id')
    ...     def get_id(self):
    ...         return f'{self.first_name}-{self.last_name}'.lower()
    ...
    ...     def get(self, key):
    ...         return self.register.retrieve(key)()

    >>> user = User('John', 'Smith')
    >>> user.get('id')
    'john-smith'
    '''

    def instantiate(self, obj):
        register = Register()
        for key, value in self.items():
            if hasattr(obj, value.__name__):
                register(key, getattr(obj, value.__name__))
            else:
                register(key, value)
        return register


class LazyRegisterValue:
    def __init__(self, register, identifier):
        self.register = register
        self.identifier = identifier
        self._registered = NO_VALUE

    def get(self):
        if self._registered is NO_VALUE:
            _registered = self.register.memory.get(self.identifier, NO_VALUE)
            if _registered is NO_VALUE:
                raise Exception(Register._msg_not_registered(self.identifier))
            else:
                self._registered = _registered
        return self._registered

    def __call__(self, *args, **kwargs):
        return self.get()(*args, **kwargs)
