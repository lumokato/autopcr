import os
from os import PathLike

from ..constants import RESULT_DIR


def result_filename(path: str) -> str:
    return path.replace("\\", "/").rsplit("/", 1)[-1]


def resolve_result_path(path: str, result_dir: str | PathLike[str] = RESULT_DIR) -> str:
    if not path:
        return ""
    if os.path.isabs(path) and os.path.isfile(path):
        return path
    return os.path.join(result_dir, result_filename(path))
