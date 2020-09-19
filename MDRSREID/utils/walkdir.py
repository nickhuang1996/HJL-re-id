import os


def walkdir(folder, exts=None, sub_path=False, abs_path=False):
    """Walk through each files in a directory.
    Reference: https://github.com/tqdm/tqdm/wiki/How-to-make-a-great-Progress-Bar
    Args:
        exts: file extensions, e.g. '.jpg', or ['.jpg'] or ['.jpg', '.png']
        sub_path: whether to exclude `folder` in the resulting paths, remaining sub paths
        abs_path: whether to return absolute paths
    """
    if isinstance(exts, str):
        exts = [exts]
    for dirpath, dirs, files in os.walk(folder):
        for filename in files:
            if (exts is None) or (os.path.splitext(filename)[1] in exts):
                path = os.path.join(dirpath, filename)
                if sub_path:
                    path = path[len(folder) + 1:]
                elif abs_path:
                    path = os.path.abspath(path)
                yield path
