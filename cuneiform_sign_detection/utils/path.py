import os
import shutil
from pathlib import Path
from typing import Union


def log_version_increment(path: Path) -> int:
    try:
        versions = [int(dir.name) for dir in path.iterdir() if dir.name.isdigit()]
        if len(versions):
            return max(versions) + 1
        else:
            return 1
    except FileNotFoundError:
        return 1


def create_directory(path: Union[str, Path], overwrite: bool = False) -> None:
    if overwrite:
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)
    else:
        os.makedirs(path)
