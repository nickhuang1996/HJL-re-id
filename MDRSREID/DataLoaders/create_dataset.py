from .Datasets.market1501 import Market1501
from .Datasets.duke import DukeMTMCreID
from .Datasets.cuhk03_np_detected_jpg import CUHK03NpDetectedJpg
from .Datasets.cuhk03_np_detected_png import CUHK03NpDetectedPng

# Use factory mode to create the different dataset
dataset_factory = {
        'market1501': Market1501,
        'cuhk03_np_detected_png': CUHK03NpDetectedPng,
        'cuhk03_np_detected_jpg': CUHK03NpDetectedJpg,
        'duke': DukeMTMCreID,
        # 'coco': COCO,
        # 'msmt17': MSMT17,
        # 'partial_reid': PartialREID,
        # 'partial_ilids': PartialiLIDs,
    }

# the shortcut of the dataset
dataset_shortcut = {
    'market1501': 'M',
    'cuhk03_np_detected_png': 'C',
    'cuhk03_np_detected_jpg': 'C',
    'duke': 'D',
    # 'msmt17': 'MS',
    # 'partial_reid': 'PR',
    # 'partial_ilids': 'PI',
}


def create_dataset(cfg,
                   mode=None,
                   domain=None,
                   name=None,
                   authority=None,
                   train_type=None,
                   items=None):
    if name not in list(dataset_factory.keys()):
        raise KeyError("Invalid dataset, got '{}', but expected to be one of {}".format(name, list(dataset_factory.keys())))
    return dataset_factory[name](cfg,
                                 mode=mode,
                                 domain=domain,
                                 name=name,
                                 authority=authority,
                                 train_type=train_type,
                                 items=items)
