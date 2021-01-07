
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
        return representation

    def value_repr(self):
        return repr(self.value)

    def repr(self, lisp_style, enable_prev):
        representation = self.value_repr()
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
        tree = self.__class__(value, False, self)
        tree.opened = True
        return tree

    def reduce(self, value=None):
        tree = self  # reducing itself possible
        children = []
        while tree.is_closed():
            children.append(tree)
            tree = tree.prev

        if not value:
            value = tree.value

        new_tree = self.__class__(value, False, tree.prev)
        new_tree.opened = False
        new_tree.children = tuple(reversed(children))

        return new_tree

    def get_parent_siblings(self):
        tree = self.prev  # not including itself
        siblings = []
        while tree.is_closed():
            siblings.append(tree)
            tree = tree.prev
        return tree, tuple(reversed(siblings))

    def get_opened_tree_children(self):
        tree = self  # including itself
        children = []
        while tree.is_closed():
            children.append(tree)
            tree = tree.prev
        return tree, tuple(reversed(children))

    def is_root(self):
        return self.prev is None

    def get_values(self):
        values = []
        self._construct_values(values)
        return values  # values don't include it's parent

    def _construct_values(self, values):
        values.append(self.value)
        if not self.terminal:
            for child in self.children:
                child._construct_values(values)

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
