import re


def overwrite_config_file(dst_config_path, ow_str='None', ow_file='None', new_cfg_file='None'):
    """Overwrite some items of a EasyDict defined config file.
    Args:
        dst_config_path: The original config file
        ow_str: Mutually exclusive to ow_file. Specify the new items (separated by ';') to overwrite.
            E.g. "cfg.model = 'ResNet-50'; cfg.im_mean = (0.5, 0.5, 0.5)".
        ow_file: A text file, each line being a new item.
        new_cfg_file: Where to write the updated config. If 'None', overwrite the original file.
    """
    with open(dst_config_path, 'r') as f:
        lines = f.readlines()
    if ow_str != 'None':
        cfgs = ow_str.split(';')
        cfgs = [cfg.strip() for cfg in cfgs if cfg.strip()]
    else:
        with open(ow_file, 'r') as f:
            cfgs = f.readlines()
        # Skip empty or comment lines
        cfgs = [cfg.strip() for cfg in cfgs if cfg.strip() and not cfg.strip().startswith('#')]
    for cfg in cfgs:
        key, value = cfg.split('=')
        key = key.strip()
        value = value.strip()
        pattern = r'{}\s*=\s*(.*?)(\s*)(#.*)?(\n|$)'.format(key.replace('.', '\.'))
        def func(x):
            # print(r'=====> {} groups, x.groups(): {}'.format(len(x.groups()), x.groups()))
            # x.group(index), index starts from 1
            # x.group(index) may be `None`
            # x.group(4) is either '\n' or ''
            return '{} = {}'.format(key, value) + (x.group(2) or '') + (x.group(3) or '') + x.group(4)
        new_lines = []
        for line in lines:
            # Skip empty or comment lines
            if not line.strip() or line.strip().startswith('#'):
                new_lines.append(line)
                continue
            line = re.sub(pattern, func, line)
            new_lines.append(line)
        lines = new_lines
    if new_cfg_file == 'None':
        new_cfg_file = dst_config_path
    with open(new_cfg_file, 'w') as f:
        # f.writelines(lines)  # Same effect
        f.write(''.join(lines))


if __name__ == '__main__':
    default_config_path = 'D:/Pycharm_Project/HJL-ReID/config/default_config.py'
    dst_config_path = 'D:/Pycharm_Project/HJL-ReID/config/dst_config.py'
    from MDRSREID.utils.src_copy_to_dst import src_copy_to_dst
    src_copy_to_dst(default_config_path, dst_config_path)

    # We should change the destination config, not the default one.
    ow_config_path = 'D:/Pycharm_Project/HJL-ReID/config/PCB_config.txt'
    # choose the overwrite content file
    overwrite_config_file(dst_config_path, ow_file=ow_config_path)
    ow_str = 'cfg.dataset.train.name = \'market\''
    # choose the overwrite str
    overwrite_config_file(dst_config_path, ow_str=ow_str)