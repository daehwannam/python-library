
import sys
import os
import io
import json
import pickle
import logging
import pprint
import pathlib
import tempfile
import shutil
import glob
import re
import itertools
import fractions


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


def get_octal_mode(file_path):
    '''
    Example:

    >>> get_octal_mode('/tmp')  # doctest: +SKIP
    '777'                       # doctest: +SKIP
    '''

    # This code is copied from
    # https://www.geeksforgeeks.org/how-to-get-the-permission-mask-of-a-file-in-python/

    return oct(os.stat(file_path).st_mode)[-3:]


def set_octal_mode(file_path, mode: str):
    '''
    Example:

    >>> file_path = '/tmp/temp-file'      # doctest: +SKIP
    >>> with open(file_path, 'w'):        # doctest: +SKIP
    ...     pass                          # doctest: +SKIP
    ...                                   # doctest: +SKIP
    >>> set_octal_mode(file_path, '777')  # doctest: +SKIP
    >>> get_octal_mode(file_path)         # doctest: +SKIP
    '777'                                 # doctest: +SKIP
    '''
    assert isinstance(mode, str)
    octal_code = int(mode, 8)
    os.chmod(file_path, octal_code)


def change_extension(path, new_ext):
    pre, ext = os.path.splitext(path)
    return '{}.{}'.format(pre, new_ext)


def open_with_mkpdirs(path, *args):
    mkpdirs_unless_exist(path)
    return open(path, *args)


def _make_dirs_unless_exist(path, to_dir):
    if to_dir:
        dir_path = os.path.dirname(os.path.join(path, ''))
    else:
        dir_path = os.path.dirname(path)
    if not os.path.isdir(dir_path):
        assert not os.path.isfile(dir_path)
        os.makedirs(dir_path)


def mkpdirs_unless_exist(path):
    "Make parent directories of a file"
    _make_dirs_unless_exist(path, to_dir=False)


def mkloc_unless_exist(path):
    "Make directories (for a location) in the path to a directory"
    _make_dirs_unless_exist(path, to_dir=True)


def touch(path):
    if not os.path.isfile(path):
        with open(path, 'w') as f:
            pass


def touch_with_mkpdirs(path):
    mkpdirs_unless_exist(path)
    touch(path)


def get_parent_path(path, depth=1):
    assert depth >= 1
    return pathlib.Path(path).parents[depth - 1].as_posix()


class ExtendedJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, set):
            return {'__py__set': list(obj)}
        elif isinstance(obj, fractions.Fraction):
            return {'__py__Fraction': [obj.numerator, obj.denominator]}
        else:
            return super().default(obj)


def as_python_object_from_json(dic):
    # https://stackoverflow.com/a/8230373
    if '__py__set' in dic:
        return set(dic['__py__set'])
    elif '__py__Fraction' in dic:
        return fractions.Fraction(*dic['__py__Fraction'])
    else:
        return dic


# class ExtendedJSONDecoder(json.JSONDecoder):
#     def default(self, obj):
#         if isinstance(obj, dict) and '__py__set' in obj:
#             return set(obj['__py__set'])
#         else:
#             return super().default(obj)


def json_skip_types(*types):
    '''
    Example:

    >>> data = [1, 2, 3, {'some_key': 'some_value'}, set(['an', 'set', 'example']), bytes([0, 1, 2, 3])]
    >>> j = json.dumps(data, cls=json_skip_types(set, bytes))
    >>> print(json.loads(j))
    [1, 2, 3, {'some_key': 'some_value'}, "SKIP: <class 'set'>", "SKIP: <class 'bytes'>"]
    '''

    if len(types) == 1 and not isinstance(types[0], type):
        types = types[0]

    for typ in types:
        assert isinstance(typ, type)
    if isinstance(types, tuple):
        types = tuple(types)

    class SkippingJSONEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, types):
                return f'SKIP: {str(type(obj))}'
            else:
                return super().default(obj)

    return SkippingJSONEncoder


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


class NoLogger:
    def do_nothing(self, *args, **kwargs):
        pass

    debug = info = warning = error = critical = exception = log = do_nothing


class _ReplaceDirectory:
    def __init__(self, dir_path, strict=True):
        self.dir_path = dir_path
        self.strict = strict

    def __enter__(self):
        if not os.path.isdir(self.dir_path):
            if os.path.isfile(self.dir_path):
                raise Exception(f'"{self.dir_path}" is a file rather than a directory')
            elif self.strict:
                raise Exception(f'"{self.dir_path}" does not exist')
            else:
                os.makedirs(self.dir_path)
        parent_dir_path = get_parent_path(self.dir_path)
        self.temp_dir_path = tempfile.mkdtemp(dir=parent_dir_path)
        dir_octal_mode = get_octal_mode(self.dir_path)
        set_octal_mode(self.temp_dir_path, dir_octal_mode)
        return self.temp_dir_path

    def __exit__(self, except_type, except_value, except_traceback):
        if (except_type, except_value, except_traceback) != (None, None, None):
            return False

        shutil.rmtree(self.dir_path)
        os.rename(self.temp_dir_path, self.dir_path)


def replace_dir(dir_path, strict=True):
    '''
    :param dir_path: The path to a directory
    :param strict: If strict=False, an empty directory to dir_path is created when the directory of dir_path does not exist.
        Otherwise raise exception.

    Example:

    >>> dir_path = 'some-dir'
    >>> os.makedirs(dir_path)
    >>> with open(os.path.join(dir_path, 'some-file-1'), 'w') as f:
    ...     pass
    ...
    >>> os.listdir(dir_path)
    ['some-file-1']
    >>> with replace_dir(dir_path) as temp_dir_path:
    ...     with open(os.path.join(temp_dir_path, 'some-file-2'), 'w') as f:
    ...         pass
    ...
    >>> os.listdir(dir_path)
    ['some-file-2']
    >>> shutil.rmtree(dir_path)  # remove the directory
    '''
    return _ReplaceDirectory(dir_path, strict=strict)


class _PrepareDirectory:
    def __init__(self, dir_path):
        self.dir_path = dir_path

    def __enter__(self):
        assert not os.path.exists(self.dir_path)

        parent_dir_path = get_parent_path(self.dir_path)
        mkloc_unless_exist(parent_dir_path)
        self.temp_dir_path = tempfile.mkdtemp(dir=parent_dir_path)

        return self.temp_dir_path

    def __exit__(self, except_type, except_value, except_traceback):
        if (except_type, except_value, except_traceback) != (None, None, None):
            return False

        os.rename(self.temp_dir_path, self.dir_path)


def prepare_dir(dir_path):
    '''
    :param dir_path: The path to a directory

    Example:

    >>> dir_path = 'some-dir'
    >>> with prepare_dir(dir_path) as temp_dir_path:
    ...     with open(os.path.join(temp_dir_path, 'some-file-2'), 'w') as f:
    ...         pass
    ...
    >>> os.listdir(dir_path)
    ['some-file-2']
    >>> shutil.rmtree(dir_path)  # remove the directory
    '''
    return _PrepareDirectory(dir_path)


def copy_dir(src, dst, replacing=False, overwriting=False, deep=True):
    if replacing and os.path.isdir(dst):
        shutil.rmtree(dst)

    return shutil.copytree(src, dst, dirs_exist_ok=overwriting, symlinks=not deep)


class _UpdateSymbolicLink:
    def __init__(self, src, dst, removing_old=False, strict=True):
        self.src = src
        self.dst = dst
        self.removing_old = removing_old
        self.strict = strict

    def __enter__(self):
        if not os.path.islink(self.dst):
            assert not os.path.isdir(self.dst)
            assert not os.path.isfile(self.dst)

            if self.strict:
                raise Exception(f'"{self.dst}" does not exist')
            else:
                mkpdirs_unless_exist(self.dst)

        parent_dir_path = get_parent_path(self.dst)
        self.temp_symlink_path = tempfile.mktemp(dir=parent_dir_path)
        make_symlink(self.src, self.temp_symlink_path)
        if self.strict or os.path.exists(os.path.realpath(self.dst)):
            dir_octal_mode = get_octal_mode(self.dst)
            set_octal_mode(self.temp_symlink_path, dir_octal_mode)
        return self.temp_symlink_path

    def __exit__(self, except_type, except_value, except_traceback):
        if (except_type, except_value, except_traceback) != (None, None, None):
            return False

        if self.removing_old:
            dst_real_path = os.path.realpath(self.dst)
            if os.path.exists(dst_real_path):
                remove_abspath(dst_real_path)
        if self.strict or os.path.islink(self.dst):
            os.unlink(self.dst)
        os.rename(self.temp_symlink_path, self.dst)


def update_symlink(src, dst, removing_old=False, strict=True):
    """

    :param src: the path of source
    :param dst: the path of symbolic link
    :param removing_old:
    :param strict: if strict == True, `dst` should be an already existing symbolic link

    Example:

    >>> dir_path_1 = 'some-dir-1'
    >>> os.makedirs(dir_path_1)
    >>> with open(os.path.join(dir_path_1, 'some-file-1'), 'w') as f:
    ...     pass
    ...
    >>> symlink_path = 'some-symlink'
    >>> make_symlink(dir_path_1, symlink_path)
    >>> os.listdir(symlink_path)
    ['some-file-1']
    >>> dir_path_2 = 'some-dir-2'
    >>> os.makedirs(dir_path_2)
    >>> with open(os.path.join(dir_path_2, 'some-file-2'), 'w') as f:
    ...     pass
    ...
    >>> with update_symlink(dir_path_2, symlink_path) as temp_symlink_path:
    ...     with open(os.path.join(temp_symlink_path, 'some-file-3'), 'w') as f:
    ...         pass
    ...
    >>> os.listdir(dir_path_2)
    ['some-file-2', 'some-file-3']
    >>> shutil.rmtree(dir_path_1)  # remove the directory
    >>> shutil.rmtree(dir_path_2)
    >>> os.unlink(symlink_path)

    """

    return _UpdateSymbolicLink(
        src=src,
        dst=dst,
        removing_old=removing_old,
        strict=strict
    )


def remove_abspath(abspath):
    if os.path.islink(abspath):
        os.remove(abspath)
    if os.path.isdir(abspath):
        shutil.rmtree(abspath)
    else:
        os.remove(abspath)


def get_numbers_in_path(prefix=None, suffix=None, num_type=int):
    glob_pattern = ''.join([
        '' if prefix is None else prefix,
        '*',
        '' if suffix is None else suffix
    ])
    dir_paths = glob.glob(glob_pattern)

    if len(dir_paths) > 0:
        regex_pattern = ''.join([
            r'' if prefix is None else prefix,
            r'([0-9]+)+',
            r'' if suffix is None else suffix
        ])
        regex_obj = re.compile(regex_pattern)

        def iter_test_numbers():
            for dir_path in dir_paths:
                match_obj = regex_obj.match(dir_path)
                if match_obj is not None:
                    yield num_type(match_obj.group(1))
        numbers = sorted(iter_test_numbers())
    else:
        numbers = []

    return numbers


def get_new_path_with_number(prefix=None, suffix=None, num_type=int, starting_num=0, no_first_num=False, no_first_suffix=False):
    '''
    Example:
    >>> new_path = get_new_path_with_number(   # doctest: +SKIP
            'some/common/path',                # doctest: +SKIP
            starting_num=1, no_first_num=True  # doctest: +SKIP
        )                                      # doctest: +SKIP
    # the 1st call: 'some/common/path'
    # the 2nd call: 'some/common/path1'
    # the 3rd call: 'some/common/path2'
    '''
    numbers = get_numbers_in_path(prefix=prefix, suffix=suffix, num_type=num_type)
    if len(numbers) > 0:
        new_number_str = str(max(numbers) + 1)
    else:
        if no_first_num:
            new_number_str = ''
        else:
            new_number_str = str(starting_num)

        if no_first_suffix:
            suffix = ''

    new_path = ''.join([
        '' if prefix is None else prefix,
        new_number_str,
        '' if suffix is None else suffix
    ])
    return new_path


def copy_matched(glob_pattern, destination):
    '''
    Example:

    >>> copy_matched('some/path/to/*', 'some/path/to/destination')  # doctest: +SKIP
    '''
    paths = glob.glob(glob_pattern)
    for path in paths:
        shutil.copy(path, destination)


def asserts_conditional_exist(path, existing):
    if existing:
        assert os.path.exists(path), f'The following path does not exist: {path}'
    else:
        assert not os.path.exists(path), f'The following path already exists: {path}'
    return path


def asserts_exist(path):
    return asserts_conditional_exist(path, True)


def asserts_not_exist(path):
    return asserts_conditional_exist(path, False)


def make_symlink(src, dst):
    """
    Make a symlink with the relative path to a destination.

    :param src: The path to the src file
    :param dst: The path to the symbolic link that indicates the src file
    """

    relative_path_to_src_from_dst = os.path.relpath(src, os.path.dirname(dst))
    os.symlink(relative_path_to_src_from_dst, dst)


def change_symlink(src, dst, removing_old=False, strict=True):
    if os.path.islink(dst):
        if removing_old:
            dst_real_path = os.path.realpath(dst)
            if os.path.exists(dst_real_path):
                remove_abspath(dst_real_path)
        os.unlink(dst)
    elif strict:
        raise Exception('{dst} is not a symbolic link')

    make_symlink(src, dst)


def copy_symlink(src, dst, replacing=False, removing_old=False):
    """
    Copy the symbolic link of the path of `src` to the path of `dst`
    """

    assert os.path.islink(src)

    if removing_old and os.path.islink(dst):
        old_dst_realpath = os.path.realpath(dst)
    else:
        old_dst_realpath = None

    if replacing and os.path.islink(dst):
        os.unlink(dst)

    # Copying a symbolic link
    # https://stackoverflow.com/a/4847660
    linkto = os.path.join(os.path.dirname(src), os.readlink(src))
    make_symlink(linkto, dst)

    if old_dst_realpath is not None:
        remove_abspath(old_dst_realpath)


def is_same_realpath(path1, path2):
    return os.path.realpath(path1) == os.path.realpath(path2)


def is_suppath(path1, path2):
    """
    Check if `path1` is a sub-path of `path2`

    Example:

    >>> is_suppath('/tmp/a', '/tmp/a/b/c')
    True
    >>> is_suppath('/tmp/a/d', '/tmp/a/b/c')
    False
    """
    return os.path.abspath(path2).startswith(os.path.abspath(path1) + os.sep)


def get_num_matched_symlinks(glob_pattern, path, except_self=False):
    '''
    Compute the number of real paths that indicate the same real path of `path` in the glob pattern.

    Example:

    >>> get_num_matched_symlinks('some/path/to/*', 'some/path/to/source')  # doctest: +SKIP
    '''
    src_real_path = os.path.realpath(path)
    if except_self:
        src_abs_path = os.path.abspath(path)

    num_matched = 0

    symlink_path_candidates = glob.glob(glob_pattern)
    for symlink_path_candidate in symlink_path_candidates:
        if os.path.islink(symlink_path_candidate):
            if except_self and src_abs_path == os.path.abspath(symlink_path_candidate):
                continue

            dst_real_path = os.path.realpath(symlink_path_candidate)
            if src_real_path == dst_real_path:
                num_matched += 1

    return num_matched


def any_matched_symlink(glob_pattern, path, except_self=False):
    """
    Return True if a symbolic link for `glob_pattern` indicates the same location as `path` does.
    """

    return get_num_matched_symlinks(
        glob_pattern,
        path,
        except_self=except_self
    ) > 0  # when more than two symbolic links indicate the same location


def remove_obsolete_symlinks(src_glob_patterns, symlink_glob_patterns):
    checkpoint_paths = set(glob_patterns_to_paths(src_glob_patterns))
    symlink_paths = set(filter(os.path.islink, glob_patterns_to_paths(symlink_glob_patterns)))

    checkpoint_realpaths = set(map(os.path.realpath, checkpoint_paths))
    symlink_realpaths = set(map(os.path.realpath, symlink_paths))

    obsolete_checkpoint_realpaths = checkpoint_realpaths.difference(symlink_realpaths)

    for obsolete_checkpoint_realpath in obsolete_checkpoint_realpaths:
        remove_abspath(obsolete_checkpoint_realpath)


def glob_patterns_to_paths(glob_patterns):
    return (path for path in itertools.chain(*map(glob.glob, glob_patterns)))


class SymLinkManager:
    def __init__(
            self,
            src_glob_patterns,
            symlink_glob_patterns,
    ):
        self.src_glob_patterns = self._to_list(src_glob_patterns)
        self.symlink_glob_patterns = self._to_list(symlink_glob_patterns)

    @staticmethod
    def _to_list(str_or_list):
        return ([str_or_list] if isinstance(str_or_list, str) else str_or_list)

    def clean(self):
        remove_obsolete_symlinks(self.src_glob_patterns, self.symlink_glob_patterns)

    def cleaning(self, context_manager):
        return _CleanSymLinks(self, context_manager)


class _CleanSymLinks:
    def __init__(self, slm: SymLinkManager, context_manager):
        self.slm = slm
        self.context_manager = context_manager

    def __enter__(self):
        return self.context_manager.__enter__()

    def __exit__(self, except_type, except_value, except_traceback):
        return_value = self.context_manager.__exit__(except_type, except_value, except_traceback)
        self.slm.clean()
        return return_value
