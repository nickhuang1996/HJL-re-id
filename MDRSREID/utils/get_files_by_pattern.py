import os.path as osp
import glob


def get_files_by_pattern(root, pattern='a/b/*.ext', strip_root=False):
    """Optionally to only return matched sub paths."""
    # Get the abspath of each directory images.
    ret = glob.glob(osp.join(root, pattern))
    # exclude the root str, so the ret is spec['patterns']. such as ['images/train/*.jpg]
    if strip_root:
        ret = [r[len(root) + 1:] for r in ret]
    return ret
