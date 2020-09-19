import argparse


def parser_args():
    """
    :argument:
        --exp_dir
        --default_config_path
        --ow_config_path
        --ow_str
    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name',
                        type=str,
                        default='MDRS',
                        help='[Optional] Model Name for experiment directory in current directory if exp_dir is None')
    parser.add_argument('--exp_dir',
                        type=str,
                        default='D:/weights_results/HJL-ReID/HuangN_ADAM_random_erasing_margin_0.3_market_best', # 'D:/weights_results/HOReID/pre-trained',  #
                        help='[Optional] Directory to store experiment output, '
                             'including log files and model checkpoint, etc.')
    parser.add_argument('--default_config_path',
                        type=str,
                        default='D:/Pycharm_Project/HJL-ReID/MDRSREID/Settings/config/default_config.py',
                        help='A configuration file.')
    parser.add_argument('--ow_config_path',
                        type=str,
                        default='D:/Pycharm_Project/HJL-ReID/MDRSREID/Settings/config/overwrite_config/MDRS_config_ADAM_best_market1501.txt',
                        help='[Optional] A text file, each line being an item to overwrite the cfg_file.')
    parser.add_argument('--ow_str',
                        type=str,
                        default='cfg.dataset.train.name = \'market1501\'',
                        help="""[Optional] Items to overwrite the cfg_file. 
                        E.g. "cfg.dataset.train.name = \'market1501\''; cfg.model.em_dim = 256" """)
    args, _ = parser.parse_known_args()
    return args
