
import sys
import os
import io
import json
import pickle
import logging
import pprint

try:
    import jsonlines
except ModuleNotFoundError:
    pass


def get_os_independent_path(path):
    return os.path.join(*path.split('/'))


def get_relative_path_wrt_this(path):
    return os.path.join(
        os.path.abspath(os.path.dirname(__file__)),
        get_os_independent_path(path))


def get_line_gen(path, remove_newline=False):
    if isinstance(path, io.IOBase):
        f = path
    else:
        f = open(path)

    with f:
        for line in f:
            if remove_newline and line[-1] == '\n':
                line = line[:-1]
            yield line


def get_base_without_extension(path):
    file_name_only, file_extension = os.path.splitext(os.path.basename(path))
    assert file_extension[0] == '.'
    return file_name_only


def get_extension(path):
    _, file_extension = os.path.splitext(path)
    assert file_extension[0] == '.'
    return file_extension[1:]


def change_extension(path, new_ext):
    pre, ext = os.path.splitext(path)
    return '{}.{}'.format(pre, new_ext)


def open_with_mkdirs(path, *args):
    mkdirs_unless_exist(path)
    return open(path, *args)


def mkdirs_unless_exist(path, to_dir=False):
    if to_dir:
        dir_path = os.path.dirname(os.path.join(path, ''))
    else:
        dir_path = os.path.dirname(path)
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)


class ExtendedJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, set):
            return {'__py__set': list(obj)}
        else:
            return super().default(obj)


class ExtendedJSONDecoder(json.JSONDecoder):
    def default(self, obj):
        if isinstance(obj, dict) and '__py__set' in obj:
            return set(obj['__py__set'])
        else:
            return super().default(obj)


def as_python_object_from_json(dic):
    # https://stackoverflow.com/a/8230373
    if '__py__set' in dic and len(dic) == 1:
        return set(dic['__py__set'])
    else:
        return dic


def example_extended_json_encoder():
    data = [1, 2, 3, set(['an', 'set', 'example']), {'some_key': 'some_value'}]
    j = json.dumps(data, cls=ExtendedJSONEncoder)
    print(json.loads(j, object_hook=as_python_object_from_json))


def json_save(obj, path, **kwargs):
    with open(path, 'w') as f:
        json.dump(obj, f, **kwargs)


def json_pretty_save(obj, path, **kwargs):
    new_kwargs = dict(ensure_ascii=False, indent=4, sort_keys=False)
    new_kwargs.update(kwargs)
    json_save(obj, path, **new_kwargs)


def json_pretty_dump(obj, fp, **kwargs):
    new_kwargs = dict(ensure_ascii=False, indent=4, sort_keys=False)
    new_kwargs.update(kwargs)
    json.dump(obj, fp, **kwargs)


def json_load(path, **kwargs):
    with open(path) as f:
        return json.load(f, **kwargs)


def extended_json_save(obj, path, **kwargs):
    new_kwargs = dict(cls=ExtendedJSONEncoder)
    new_kwargs.update(kwargs)
    json_save(obj, path, **new_kwargs)


def extended_json_pretty_save(obj, path, **kwargs):
    new_kwargs = dict(cls=ExtendedJSONEncoder)
    new_kwargs.update(kwargs)
    json_pretty_save(obj, path, **new_kwargs)


def extended_json_load(path, **kwargs):
    new_kwargs = dict(object_hook=as_python_object_from_json)
    new_kwargs.update(kwargs)
    return json_load(path, **new_kwargs)


def pickle_save(obj, path, **kwargs):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, **kwargs)


def pickle_save_highest(obj, path, **kwargs):
    new_kwargs = dict(protocol=pickle.HIGHEST_PROTOCOL)
    new_kwargs.update(kwargs)
    pickle_save(obj, path, **new_kwargs)


def pickle_load(path, **kwargs):
    with open(path, 'rb') as f:
        return pickle.load(f, **kwargs)


def write_text(path, text):
    with open(path, 'w') as f:
        return f.write(text)


def read_text(path):
    with open(path) as f:
        return f.read()


def write_lines(path, lines):
    with open(path, 'w') as f:
        for line_num, line in enumerate(lines, 1):
            f.write(line)
            if line_num != len(lines):
                f.write('\n')


def read_lines(path):
    with open(path) as f:
        return f.readlines()


def python_save(obj, path, repr=repr):
    return write_text(path, repr(obj))


def python_pretty_save(obj, path, **kwargs):
    with open(path, 'w') as f:
        python_pretty_dump(obj, f, **kwargs)


def python_pretty_dump(obj, fp, **kwargs):
    pprint_kwargs = dict(indent=1)  # default value
    # pprint_kwargs = dict(indent=4)
    pprint_kwargs.update(kwargs)
    assert 'stream' not in pprint_kwargs

    pprint.pprint(obj, stream=fp, **pprint_kwargs)


def python_load(path, *args):
    assert len(args) <= 2  # for globals and locals
    return eval(read_text(path), *args)


def jsonl_save(objects, path, **kwargs):
    with jsonlines.open(path, mode='w') as writer:
        for obj in objects:
            writer.write(obj)

def jsonl_load(path, **kwargs):
    with jsonlines.open(path) as reader:
        return tuple(reader)


def make_logger(name, log_file_path=None, to_stdout=True, overwriting=False, format_str=None, level=None):
    if format_str is None:
        log_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    else:
        log_formatter = logging.Formatter(format_str)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG if level is None else level)
    # logging.DEBUG is the lowest level.
    # When the level is set as logging.DEBUG, the logger prints all messages.

    if log_file_path is not None:
        mode = 'w' if overwriting else 'a'
        file_handler = logging.FileHandler("{0}".format(log_file_path), mode=mode)
        file_handler.setFormatter(log_formatter)
        logger.addHandler(file_handler)

    if to_stdout:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(log_formatter)
        logger.addHandler(console_handler)

    return logger
