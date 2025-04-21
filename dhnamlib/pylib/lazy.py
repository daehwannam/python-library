
from abc import ABCMeta, abstractmethod

from .klass import subclass, implement, override
from .decoration import deprecated, curry
from .function import identity
from .constant import NO_VALUE


class LazyObject(metaclass=ABCMeta):
    @abstractmethod
    def _get_evaluated_obj(self):
        pass


@subclass
class LazyEval(LazyObject):
    def __init__(self, fn, *args, **kwargs):
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.evaluated = False
        self.obj = NO_VALUE

    @implement
    def _get_evaluated_obj(self):
        if not self.evaluated:
            self._evaluate_and_set_obj()

        return self.obj

    def _evaluate_obj(self):
        self._evaluate_and_set_obj()
        return self.obj

    def _evaluate_and_set_obj(self):
        self.obj = self.fn(*self.args, **self.kwargs)
        self.evaluated = True


def eval_lazy_obj(obj, recursive=False, lazy_object_cls=LazyObject):
    while isinstance(obj, lazy_object_cls):
        obj = obj._get_evaluated_obj()
        if not recursive:
            break
    return obj


def eval_lazy_obj_recursively(obj, lazy_object_cls=LazyObject):
    return eval_lazy_obj(obj, recursive=True, lazy_object_cls=lazy_object_cls)



@deprecated
def eval_obj_unless_lazy(obj, lazy):
    if lazy:
        return obj
    else:
        return eval_lazy_obj(obj)


@deprecated
def get_eval_obj_unless_lazy(lazy):
    if lazy:
        return identity
    else:
        return eval_lazy_obj


@subclass
class LazyProxy(LazyObject):
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

    def _initialize(self, fn, *args, **kwargs):
        # super().__setattr__('_lazy_obj', LazyEval(fn, *args, **kwargs))
        self._lazy_obj = LazyEval(fn, *args, **kwargs)
        self._evaluated_obj = NO_VALUE

    def __init__(self, fn, *args, **kwargs):
        self._initialize(fn, *args, **kwargs)
        self.__setattr__ = self._setattr

    @implement
    def _get_evaluated_obj(self):
        # return self._lazy_obj._get_evaluated_obj()
        # return eval_lazy_obj(self.lazy_obj, recursive=True)
        if self._evaluated_obj is NO_VALUE:
            _evaluated_obj = eval_lazy_obj(
                self._lazy_obj._get_evaluated_obj(),
                recursive=True,
                lazy_object_cls=LazyProxy
            )
            super().__setattr__('_evaluated_obj', _evaluated_obj)
        return self._evaluated_obj

    def __getattr__(self, name):
        return getattr(self._get_evaluated_obj(), name)

    # def __setattr__(self, name, value):
    #     setattr(self._get_evaluated_obj(), name, value)

    def _setattr(self, name, value):
        setattr(self._get_evaluated_obj(), name, value)

    def __call__(self, *args, **kwargs):
        return self._get_evaluated_obj().__call__(*args, **kwargs)

    def __repr__(self):
        # class_name = LazyProxy.__module__ + '.' + LazyProxy.__qualname__
        return '{}({})'.format(self.__class__.__name__, repr(self._get_evaluated_obj()))

    def __getitem__(self, key):
        return self._get_evaluated_obj().__getitem__(key)

    def __setitem__(self, key, value):
        self._get_evaluated_obj().__setitem__(key, value)


@subclass
class DynamicLazyProxy(LazyProxy):
    '''
    It's same with LazyProxy except `_get_evaluated_obj` always evaluates the LazyEval object.
    '''

    @override
    def _initialize(self, fn, *args, **kwargs):
        # super().__setattr__('_lazy_obj', LazyEval(fn, *args, **kwargs))
        self._lazy_obj = LazyEval(fn, *args, **kwargs)
        # self._evaluated_obj = NO_VALUE

    @override
    def _get_evaluated_obj(self):
        _evaluated_obj = eval_lazy_obj(
            self._lazy_obj._get_evaluated_obj(),
            recursive=True,
            lazy_object_cls=LazyProxy
        )
        return _evaluated_obj


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

    def retrieve_instantly(self, identifier):
        return self.retrieve(identifier, strategy='instant')

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


@subclass
class LazyRegisterValue(LazyProxy):
    def _initialize(self, register, identifier):
        self._register = register
        self._identifier = identifier
        self._lazy_obj = LazyEval(self.__get_registered_obj)
        self._evaluated_obj = NO_VALUE

    def __init__(self, register, identifier):
        self._initialize(register, identifier)
        self.__setattr__ = self._setattr

    def __get_registered_obj(self):
        _registered = self._register.memory.get(self._identifier, NO_VALUE)
        if _registered is NO_VALUE:
            raise Exception(Register._msg_not_registered(self._identifier))
        return _registered
