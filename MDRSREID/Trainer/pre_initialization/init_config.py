from MDRSREID.Settings.parser_args.parser_args import parser_args
import os.path as osp
from MDRSREID.utils.src_copy_to_dst import src_copy_to_dst
from MDRSREID.utils.config_utils.overwrite_config_file import overwrite_config_file
from MDRSREID.utils.import_file import import_file


def init_config(args=None):
    """
    args can be parsed from command line, or provided by function caller.

    Load the args.
    Set the experiment directory.
    Copy default config file to dst file.
    Overwrite the contents.
    """
    if args is None:
        args = parser_args()

    # Set the experiment directory
    exp_dir = args.exp_dir
    if exp_dir is None:
        exp_dir = 'experiment/' + args.model_name + '/' + osp.splitext(osp.basename(args.default_config_path))[0]

    # copy file
    dst_config_path = osp.join(exp_dir, osp.basename(args.default_config_path))
    src_copy_to_dst(args.default_config_path, dst_config_path)

    # overwrite
    if args.ow_config_path != 'None':
        print('ow_config_path is: {}'.format(args.ow_config_path))
        overwrite_config_file(dst_config_path, ow_file=args.ow_config_path)
    if args.ow_str != 'None':
        print('ow_str is: {}'.format(args.ow_str))
        overwrite_config_file(dst_config_path, ow_str=args.ow_str)

    # import config
    cfg = import_file(dst_config_path).cfg

    # Set log experiment dir
    cfg.log.exp_dir = exp_dir
    return cfg


if __name__ == '__main__':
    cfg = init_config()
