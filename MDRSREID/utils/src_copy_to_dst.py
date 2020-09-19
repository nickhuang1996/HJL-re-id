import os.path as osp
import shutil
from MDRSREID.utils.may_make_dirs import may_make_dirs


def src_copy_to_dst(src_path, dst_path):
    """
    :param src_path:
    :param dst_path:
    :return:

    We may make destination path.
    Then we will copy files.
    """
    if osp.exists(src_path):
        may_make_dirs(dst_path=osp.dirname(dst_path))
    shutil.copy(src_path, dst_path)


if __name__ == '__main__':
    src_path = 'a/test.txt'
    dst_path = 'b/test2.txt'
    src_copy_to_dst(src_path, dst_path)