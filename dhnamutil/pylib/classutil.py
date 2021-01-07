
def get_all_subclass_set(cls):
    subclass_list = []

    def recurse(klass):
        for subclass in klass.__subclasses__():
            subclass_list.append(subclass)
            recurse(subclass)

    recurse(cls)

    return set(subclass_list)


def find_unique_subclass_in_module(superclass, module):
    subclass = None
    for obj_name in dir(module):
        klass = getattr(module, obj_name)
        if klass != superclass and \
           isinstance(type(klass), type) and \
           isinstance(klass, type) and \
           issubclass(klass, superclass):
            if subclass is None:
                subclass = klass
            else:
                raise Exception("No more than 1 subclass of {} is allowed.".format(superclass.__name__))
    if subclass is None:
        raise Exception("No subclass of {} exists".format(superclass.__name__))

    return subclass
