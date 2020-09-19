from MDRSREID.Networks import Model
from MDRSREID.utils.device_utils.may_data_parallel import may_data_parallel


def model_creation(cfg):
    """
    :param cfg:
    :return: model

    Init the model.
    May put the model on gpu(s).
    """
    model = Model(cfg)
    # may use not only one gpu.
    model = may_data_parallel(model)
    model.to(cfg.device)
    print(model)
    num_param = model.count_num_param()
    print("Model size: {:.3f} M".format(num_param))
    return model
