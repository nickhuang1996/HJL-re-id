from MDRSREID.Networks.RESIDUAL_NETWORK.POSE_HRNET.PoseHighResolutionNet import get_pose_net


keypoints_predictor_factory = {
    'pose_hrnet': get_pose_net
}


def create_keypoints_predictor(cfg):
    return keypoints_predictor_factory[cfg.keypoints_model.name](cfg)


if __name__ == '__main__':
    from MDRSREID.Trainer.pre_initialization.init_config import init_config
    import torch

    print("[PYTORCH VERSION]:", torch.__version__)
    cfg = init_config()
    print("Keypoints model Name:{}\n".format(cfg.keypoints_model.name))
    create_keypoints_predictor(cfg)
