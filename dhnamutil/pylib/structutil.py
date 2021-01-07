
import re
from .typeutil import isanyinstance


class AttrDict(dict):
    # https://stackoverflow.com/a/14620633
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

    def rename_attr(self, old_name, new_name):
        assert new_name not in self
        self[new_name] = self[old_name]
        del self[old_name]

    # def get_attr_gen(self, *attr_keys):
    #     # if len(attr_keys) == 0:
    #     #     return ()
    #     if len(attr_keys) == 1:
    #         if isinstance(attr_keys[0], str):
    #             attr_keys = attr_keys[0].replace(',', '').split()
    #         else:
    #             attr_keys = attr_keys[0]  # attr_keys[0] is a sequence such as list or tuple
    #     for attr in attr_keys:
    #         yield self[attr]


def get_recursive_attr_dict(obj):
    if isinstance(obj, AttrDict):
        return obj
    elif isinstance(obj, dict):
        return AttrDict(
            (k, get_recursive_attr_dict(v))
            for k, v in obj.items())
    elif any(isinstance(obj, typ) for typ in [list, tuple, set]):
        return type(obj)(get_recursive_attr_dict(elem) for elem in obj)
    else:
        return obj


def get_recursive_dict(obj, dict_cls=dict):
    if issubclass(type(dict_cls), type) and issubclass(dict_cls, dict) and type(obj) == dict_cls:
        return obj
    elif isinstance(obj, dict):
        return dict_cls(
            (k, get_recursive_attr_dict(v))
            for k, v in obj.items())
    elif isinstance(obj, (list, tuple, set)):
        return type(obj)(get_recursive_attr_dict(elem) for elem in obj)
    else:
        return obj


def copy_recursively(obj):
    if isanyinstance(obj, [list, tuple, set]):
        return type(obj)(map(copy_recursively, obj))
    elif isinstance(obj, dict):
        return type(obj)([copy_recursively(k), copy_recursively(v)]
                         for k, v in obj.items())
    else:
        return obj


class TreeStructure:
    @classmethod
    def create_root(cls, value, terminal=False):
        struct = cls(value, terminal, None)
        struct.opened = not terminal
        return struct

    def __init__(self, value, terminal, prev):
        self.value = value
        self.terminal = terminal
        self.prev = prev

    def __repr__(self, lisp_style=True, enable_prev=True):
        representation = self.repr(
            lisp_style=lisp_style, enable_prev=enable_prev)
        # if lisp_style:
        #     representation = re.sub(
        #         r'lambda_(.)',
        #         lambda x: 'lambda ({})'.format(x.group(1)),
        #         representation)
        # representation = re.sub(
        #     ' TRUE', '', representation, flags=re.IGNORECASE)
        return representation

    def repr(self, lisp_style, enable_prev, symbol_repr=False):
        representation = str(self.value)  # or repr(self.value)
        if symbol_repr:
            representation = camel_to_symbol(representation)
        if not self.terminal:
            representation = '(' + representation + ' ' if lisp_style else \
                             representation + '('
            if not self.opened:
                delimiter = ' ' if lisp_style else ', '
                representation = representation + \
                    delimiter.join(child.repr(lisp_style=lisp_style, enable_prev=False)
                                   for child in self.children) + ')'
        if self.prev and enable_prev:
            if self.prev.is_closed():
                delimiter = ' ' if lisp_style else ', '
                representation = '{}{}{}'. format(
                    self.prev.repr(lisp_style=lisp_style, enable_prev=True),
                    delimiter, representation)
            else:
                representation = '{}{}'. format(
                    self.prev.repr(lisp_style=lisp_style, enable_prev=True),
                    representation)

        return representation

    def is_opened(self):
        return not self.terminal and self.opened

    def is_closed(self):
        return self.terminal or not self.opened

    def push_term(self, value):
        return self.__class__(value, True, self)

    def push_nonterm(self, value):
        program = self.__class__(value, False, self)
        program.opened = True
        return program

    # def is_opened(self):
    #     return not self.terminal and self.opened

    def reduce(self, value=None):
        program = self  # reducing itself possible
        children = []
        while program.is_closed():
            children.append(program)
            program = program.prev

        if not value:
            value = program.value

        new_program = self.__class__(value, False, program.prev)
        new_program.opened = False
        new_program.children = tuple(reversed(children))

        return new_program

    def get_parent_siblings(self):
        program = self.prev
        children = []
        while program.is_closed():
            children.append(program)
            program = program.prev
        return program, tuple(reversed(children))

    def get_opened_program_children(self):
        program = self
        children = []
        while program.is_closed():
            children.append(program)
            program = program.prev
        return program, tuple(reversed(children))

    def is_root(self):
        return self.prev is None

    # def is_closed_root(self):
    #     return self.is_closed() and self.is_root()

    def is_complete(self):  # == is_closed_root
        return self.is_root() and self.is_closed()

    def get_values(self):
        values = []
        self._construct_values(values)
        return values  # values don't include it's parent

    def _construct_values(self, values):
        values.append(self.value)
        if not self.terminal:
            for child in self.children:
                child._construct_values(values)

    def get_all_values(self):
        def get_values(program):
            if program.is_closed():
                return program.get_values()
            else:
                return [program.value]

        all_values = []

        def recurse(program):
            if not program.is_root():
                parent, siblings = program.get_parent_siblings()
                recurse(parent)
                for sibling in siblings:
                    all_values.extend(get_values(sibling))
            all_values.extend(get_values(program))

        recurse(self)
        return all_values

    def get_num_tokens(self, enable_prev=True):
        count = 1
        if enable_prev and self.prev:
            count += self.prev.get_num_tokens()
        if not self.terminal and not self.opened:
            for child in self.children:
                count += child.get_num_tokens(False)
        return count

    def get_last_value(self):
        if self.terminal or self.opened:
            return self.value
        else:
            return self.children[-1].get_last_value()

    def find_sub_tree(self, item, key=lambda x: x.value):
        # don't consider 'prev'
        if key(self) == item:
            return self
        elif self.terminal:
            return None
        else:
            for child in self.children:
                sub_tree = child.find_sub_tree(item, key)
                if sub_tree:
                    return sub_tree
            else:
                return None


first_cap_re = re.compile('(.)([A-Z][a-z]+)')
all_cap_re = re.compile('([a-z0-9])([A-Z])')


def camel_to_symbol(name):
    s1 = first_cap_re.sub(r'\1-\2', name)
    return all_cap_re.sub(r'\1-\2', s1).lower()


def camel_to_snake(name):
    s1 = first_cap_re.sub(r'\1-\2', name)
    return all_cap_re.sub(r'\1-\2', s1).lower()
