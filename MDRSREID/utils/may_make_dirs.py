import os
import os.path as osp
import sys


def may_make_dirs(STD=None, dst_path=None):
    if STD.__class__.__name__ != 'ReDirectSTD':
        STD = sys.stdout
        if dst_path in [None, '']:
            # print("destination path is None!!")
            STD.write("destination path dir is None!!\n")
            STD.flush()
            return
        elif not osp.exists(dst_path):
            os.makedirs(dst_path)
            # print("destination path" + dst_path + "has been created.")
            STD.write(str("destination path dir " + dst_path + " has been created.\n"))
            STD.flush()
        else:
            # print("destination path" + dst_path + "has been already exist.")
            STD.write(str("destination path dir " + dst_path + " has been already exist.\n"))
            STD.flush()
            return
    else:
        if dst_path in [None, '']:
            return
        elif not osp.exists(dst_path):
            os.makedirs(dst_path)
        else:
            return
