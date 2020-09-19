
def import_file(path):
    import sys, importlib
    import os.path as osp
    path_to_insert = osp.dirname(osp.abspath(osp.expanduser(path)))
    sys.path.insert(0, path_to_insert)
    imported = importlib.import_module(osp.splitext(osp.basename(path))[0])
    # NOTE: sys.path may be modified inside the imported file. If path_to_insert
    # is not added to sys.path at any index inside the imported file, this remove()
    # can exactly cancel the previous insert(). Otherwise, the element order inside
    # sys.path may not be desired by the imported file.
    sys.path.remove(path_to_insert)
    return imported
