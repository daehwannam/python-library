
import subprocess


def get_git_hash(cwd=None, short=False) -> str:
    '''
    Retrieve SHA-1 hash value of a git repository.
    `cwd` stands for the current working directory, which should be a path to a sub-directory of the git repository.
    '''

    # This code is copied from
    # https://stackoverflow.com/a/21901260

    if short:
        cmd_args = ['git', 'rev-parse', '--short', 'HEAD']
    else:
        cmd_args = ['git', 'rev-parse', 'HEAD']

    return subprocess.check_output(cmd_args, cwd=cwd).decode('ascii').strip()
