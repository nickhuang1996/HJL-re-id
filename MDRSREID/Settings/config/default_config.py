"""This is the config for GlobalPool.
Some config may be reconfigured and new items may be added at run time."""

from __future__ import print_function
from easydict import EasyDict

cfg = EasyDict()
# ===========================================keypoint model config=========================================== #


cfg.keypoints_model = EasyDict()
cfg.keypoints_model.test = EasyDict()

# model extras
pose_resnet = EasyDict()

pose_high_resolution_net = EasyDict()
pose_high_resolution_net.pretrained_layers = ['*']
pose_high_resolution_net.stem_inplanes = 64

# stage 2
pose_high_resolution_net.stage2 = EasyDict()
pose_high_resolution_net.stage2.num_modules = 1
pose_high_resolution_net.stage2.num_branches = 2
pose_high_resolution_net.stage2.num_blocks = [4, 4]
pose_high_resolution_net.stage2.num_channels = [48, 96]
pose_high_resolution_net.stage2.block = 'BASIC'
pose_high_resolution_net.stage2.fuse_method = 'SUM'

# stage 3
pose_high_resolution_net.stage3 = EasyDict()
pose_high_resolution_net.stage3.num_modules = 4
pose_high_resolution_net.stage3.num_branches = 3
pose_high_resolution_net.stage3.num_blocks = [4, 4, 4]
pose_high_resolution_net.stage3.num_channels = [48, 96, 192]
pose_high_resolution_net.stage3.block = 'BASIC'
pose_high_resolution_net.stage3.fuse_method = 'SUM'

# stage 4
pose_high_resolution_net.stage4 = EasyDict()
pose_high_resolution_net.stage4.num_modules = 3
pose_high_resolution_net.stage4.num_branches = 4
pose_high_resolution_net.stage4.num_blocks = [4, 4, 4, 4]
pose_high_resolution_net.stage4.num_channels = [48, 96, 192, 384]
pose_high_resolution_net.stage4.block = 'BASIC'
pose_high_resolution_net.stage4.fuse_method = 'SUM'

pose_high_resolution_net.final_conv_kernel = 1


model_extras = {
    'pose_resnet': pose_resnet,
    'pose_high_resolution_net': pose_high_resolution_net,
}
# pretrained(Encoder)
cfg.keypoints_model.is_train = False

cfg.keypoints_model.name = 'pose_hrnet'
cfg.keypoints_model.norm_scale = 10.0
cfg.keypoints_model.num_joints = 17
cfg.keypoints_model.extra = model_extras['pose_high_resolution_net']
cfg.keypoints_model.init_weights = True
cfg.keypoints_model.pretrained = ''

# HeatmapProcessor
cfg.keypoints_model.joints_groups = [[0, 1, 2, 3, 4],
                                     [5], [6], [7], [8], [9], [10], [11], [12], [13], [14], [15], [16]]  # coco group 2
# BNClassifiers GraphMatchingNet Verificator
cfg.keypoints_model.branch_num = 14  # 17 - 5 + 1 + 1

# LocalFeaturesComputer
cfg.keypoints_model.weight_global_feature = 1.0

cfg.keypoints_model.test.model_file = 'D:/weights_results/HOReID/pre-trained/pose_model/pose_hrnet_w48_256x192.pth'


cfg.only_test = False  #
cfg.only_infer = False  #

cfg.model = EasyDict()

cfg.model.name = 'Global'


cfg.model.backbone = EasyDict()
cfg.model.backbone.name = 'resnet50'
cfg.model.backbone.last_conv_stride = 1
cfg.model.backbone.pretrained = True
cfg.model.backbone.pretrained_model_dir = 'D:/weights_results/pretrained_model/imagenet_model'

cfg.model.RGA = EasyDict()
cfg.model.RGA.branch_name = 'rgasc'  # ['rgasc', 'rgac', 'rgas']
cfg.model.RGA.channel_ratio = 8
cfg.model.RGA.spatial_ratio = 8
cfg.model.RGA.downchannel_ratio = 8

cfg.model.pool_type = 'GlobalPool'  # ['GlobalPool', 'PCBPool', 'PAPool']
cfg.model.max_or_avg = 'max'
cfg.model.em_dim = 512  #
cfg.model.num_parts = 1  #
cfg.model.used_levels = None  #
cfg.model.use_ps = False  #

cfg.model.reduction = EasyDict()
cfg.model.reduction.use_relu = False
cfg.model.reduction.use_leakyrelu = False
cfg.model.reduction.use_dropout = False

cfg.model.PGFA = EasyDict()
cfg.model.PGFA.global_input_dim = 4096
cfg.model.PGFA.part_feat_input_dim = 2048

cfg.model.seg = EasyDict()
cfg.model.seg.out_c = 256
cfg.model.seg.num_classes = 8

cfg.model.multi_seg = EasyDict()
cfg.model.multi_seg.mid_c = 256
cfg.model.multi_seg.mid_c2 = 512
cfg.model.multi_seg.in_c1 = 2048
cfg.model.multi_seg.in_c2 = 1024
cfg.model.multi_seg.num_classes = 8

cfg.model.gcn = EasyDict()
cfg.model.gcn.scale = 20.0

cfg.dataset = EasyDict()
cfg.dataset.root = 'D:/datasets/ReID_dataset'
cfg.dataset.use_occlude_duke = False

cfg.dataset.im = EasyDict()
cfg.dataset.im.h_w = (256, 128)  # final size for network input
# https://pytorch.org/docs/master/torchvision/models.html#torchvision-models
cfg.dataset.im.mean = [0.486, 0.459, 0.408]
cfg.dataset.im.std = [0.229, 0.224, 0.225]
cfg.dataset.im.interpolation = None

cfg.dataset.im.pad = 10

cfg.dataset.im.random_erasing = EasyDict()
cfg.dataset.im.random_erasing.epsilon = 0.5
cfg.dataset.im.random_erasing.proportion = True

cfg.dataset.im.random_crop = EasyDict()
cfg.dataset.im.random_crop.output_size = (256, 128)

cfg.dataset.use_pose_landmark_mask = False  #
cfg.dataset.pose_landmark_mask = EasyDict()
cfg.dataset.pose_landmark_mask.h_w = (24, 8)  # final size for masking
cfg.dataset.pose_landmark_mask.type = 'PL_18P'

cfg.dataset.use_ps_label = False  #
cfg.dataset.ps_label = EasyDict()
cfg.dataset.ps_label.h_w = (48, 16)  # final size for calculating loss
cfg.dataset.ps_label.pad = 10 // 8  # 8 due to 128 // 16

# Note that cfg.dataset.train.* will not be accessed directly. Intended behavior e.g.
#     from package.utils.cfg import transfer_items
#     transfer_items(cfg.dataset.train, cfg.dataset)
#     print(cfg.dataset.transform_list)
# Similar for cfg.dataset.test.*, cfg.dataloader.train.*, cfg.dataloader.test.*
cfg.model_flow = 'train'  # 'train' or 'test'
cfg.stage = 'FeatureExtract'  # 'FeatureExtract' or 'Evaluation'

cfg.dataset.train = EasyDict()
cfg.dataset.train.before_to_tensor_transform_list = ['hflip', 'resize']
cfg.dataset.train.after_to_tensor_transform_list = ['random_erasing']
cfg.dataset.train.type = 'Supervised'  # 'Unsupervised'

cfg.dataset.train.source = EasyDict()
cfg.dataset.train.source.name = 'market1501'  # ['market1501', 'cuhk03_np_detected_jpg', 'duke']
cfg.dataset.train.source.authority = 'train'  #

cfg.dataset.train.target = EasyDict()
cfg.dataset.train.target.name = 'market1501'  # ['market1501', 'cuhk03_np_detected_jpg', 'duke']
cfg.dataset.train.target.authority = 'train'  #

cfg.dataset.cd_train = EasyDict()
cfg.dataset.cd_train.name = 'duke'  #
cfg.dataset.cd_train.authority = 'train'  #
cfg.dataset.cd_train.transform_list = ['hflip', 'resize']

cfg.dataset.test = EasyDict()
cfg.dataset.test.names = ['market1501', 'cuhk03_np_detected_jpg', 'duke']
if hasattr(cfg.dataset.test, 'query_authorities'):
    assert len(cfg.dataset.test.query_authorities) == len(cfg.dataset.test.names), "If cfg.dataset.test.query_authorities is defined, it should be set for each test set."
cfg.dataset.test.before_to_tensor_transform_list = ['resize']
cfg.dataset.test.after_to_tensor_transform_list = None

num_classes_dict = {
    'market1501': 751,
    'duke': 702,
    'cuhk03_np_detected_jpg': 767
}
cfg.model.num_classes = num_classes_dict[cfg.dataset.train.source.name] if cfg.only_test else None # 751 702


cfg.dataloader = EasyDict()
cfg.dataloader.num_workers = 2

cfg.dataloader.train = EasyDict()
cfg.dataloader.train.batch_type = 'random'  # 'seq'|'random'|'pk'|'pk2'
cfg.dataloader.train.batch_size = 32
cfg.dataloader.train.batch_id = 8
cfg.dataloader.train.batch_image = 4
cfg.dataloader.train.drop_last = True

cfg.dataloader.cd_train = EasyDict()
cfg.dataloader.cd_train.batch_type = 'random'
cfg.dataloader.cd_train.batch_size = 32
cfg.dataloader.cd_train.drop_last = True

cfg.dataloader.test = EasyDict()
cfg.dataloader.test.batch_type = 'seq'
cfg.dataloader.test.batch_size = 16
cfg.dataloader.test.drop_last = False

cfg.dataloader.pk = EasyDict()
cfg.dataloader.pk.k = 4

cfg.dataloader.pk2 = EasyDict()

cfg.eval = EasyDict()
cfg.eval.forward_type = 'reid'
cfg.eval.chunk_size = 1000
cfg.eval.re_rank = False
cfg.eval.separate_camera_set = False
cfg.eval.single_gallery_shot = False
cfg.eval.first_match_break = True
cfg.eval.score_prefix = ''
cfg.eval.ranked_images = False
cfg.eval.ver_in_scale = 10.0

cfg.eval.feat_dict = EasyDict()
cfg.eval.feat_dict.use = True if cfg.only_test else False
cfg.eval.feat_dict.reproduction = False

cfg.train = EasyDict()

cfg.id_loss = EasyDict()
cfg.id_loss.name = 'idL'
cfg.id_loss.weight = 1  #
cfg.id_loss.use = cfg.id_loss.weight > 0

cfg.id_smooth_loss = EasyDict()
cfg.id_smooth_loss.name = 'idSmoothL'
cfg.id_smooth_loss.weight = 0  #
cfg.id_smooth_loss.use = cfg.id_smooth_loss.weight > 0
cfg.id_smooth_loss.epsilon = 0.1
cfg.id_smooth_loss.reduce = True
cfg.id_smooth_loss.use_gpu = True

cfg.tri_loss = EasyDict()
cfg.tri_loss.name = 'triL'
cfg.tri_loss.weight = 0  #
cfg.tri_loss.use = cfg.tri_loss.weight > 0
cfg.tri_loss.margin = 1.2
cfg.tri_loss.dist_type = 'euclidean'
cfg.tri_loss.hard_type = 'tri_hard'
cfg.tri_loss.norm_by_num_of_effective_triplets = False

cfg.tri_hard_loss = EasyDict()
cfg.tri_hard_loss.name = 'triHardL'
cfg.tri_hard_loss.weight = 0
cfg.tri_hard_loss.use = cfg.tri_hard_loss.weight > 0
cfg.tri_hard_loss.margin = 0.3
cfg.tri_hard_loss.dist_type = 'euclidean'

cfg.permutation_loss = EasyDict()
cfg.permutation_loss.name = 'permutationL'
cfg.permutation_loss.weight = 0
cfg.permutation_loss.use = cfg.permutation_loss.weight > 0
cfg.permutation_loss.branch_num = cfg.keypoints_model.branch_num  # 14

cfg.verification_loss = EasyDict()
cfg.verification_loss.name = 'verL'
cfg.verification_loss.weight = 0
cfg.verification_loss.use = cfg.verification_loss.weight > 0

# Source domain ps loss
cfg.src_seg_loss = EasyDict()
cfg.src_seg_loss.name = 'sL'
cfg.src_seg_loss.weight = 0
cfg.src_seg_loss.use = cfg.src_seg_loss.weight > 0
cfg.src_seg_loss.normalize_size = True
cfg.src_seg_loss.num_classes = cfg.model.seg.num_classes

# source domain ps loss
cfg.src_multi_seg_loss = EasyDict()
cfg.src_multi_seg_loss.name = 'msL'
cfg.src_multi_seg_loss.weight = 0  #
cfg.src_multi_seg_loss.use = cfg.src_multi_seg_loss.weight > 0
cfg.src_multi_seg_loss.normalize_size = True
cfg.src_multi_seg_loss.num_classes = cfg.model.multi_seg.num_classes

# source domain psgp loss
cfg.src_multi_seg_gp_loss = EasyDict()
cfg.src_multi_seg_gp_loss.name = 'psL'
cfg.src_multi_seg_gp_loss.weight = 0  #
cfg.src_multi_seg_gp_loss.use = cfg.src_multi_seg_gp_loss.weight > 0
cfg.src_multi_seg_gp_loss.normalize_size = True
cfg.src_multi_seg_gp_loss.num_classes = cfg.model.multi_seg.num_classes

cfg.pgfa_loss = EasyDict()
cfg.pgfa_loss.name = 'pgfaL'
cfg.pgfa_loss.weight = 0
cfg.pgfa_loss.use = cfg.pgfa_loss.weight > 0
cfg.pgfa_loss.lamb = 0.2

cfg.inv_loss = EasyDict()
cfg.inv_loss.name = 'InvL'
cfg.inv_loss.num_features = 2048
cfg.inv_loss.weight = 1 #
cfg.inv_loss.use = cfg.inv_loss.weight > 0
cfg.inv_loss.beta = 0.05
cfg.inv_loss.alpha = 0.01
cfg.inv_loss.knn = 6
cfg.inv_loss.lmd = 0.3

cfg.analyze_computer = EasyDict()
cfg.analyze_computer.name = 'analyze_computer'
cfg.analyze_computer.use = False

cfg.verification_probability_analyze = EasyDict()
cfg.verification_probability_analyze.name = 'verification_probability_analyze'
cfg.verification_probability_analyze.use = False

cfg.log = EasyDict()
cfg.log.use_tensorboard = True
cfg.log.only_base_lr = False

cfg.optim = EasyDict()
cfg.optim.optimizer = 'sgd'

cfg.optim.sgd = EasyDict()
cfg.optim.sgd.momentum = 0.9
cfg.optim.sgd.nesterov = False
cfg.optim.sgd.weight_decay = 5e-4

cfg.optim.adam = EasyDict()
cfg.optim.adam.beta1 = 0.9
cfg.optim.adam.beta2 = 0.99
cfg.optim.adam.eps = 1e-8
cfg.optim.adam.amsgrad = False
cfg.optim.adam.weight_decay = 5e-4

cfg.optim.seperate_params = False
cfg.optim.base_lr = 0.00035
cfg.optim.ft_lr = 0.01  # for resume
cfg.optim.initial_ft_lr = 0.01  #
cfg.optim.new_params_lr = 0.02  # for resume
cfg.optim.initial_new_params_lr = 0.02
cfg.optim.gcn_lr_scale = 0.1
cfg.optim.gm_lr_scale = 1.0
cfg.optim.ver_lr_scale = 1.0
cfg.optim.every_lr_epoch = 40
cfg.optim.easy_epochs = 60
cfg.optim.lr_decay_epochs = (25, 50)  #
cfg.optim.normal_epochs = 60  # Not including warmup/pretrain
cfg.optim.warmup_epochs = 0
cfg.optim.warmup = cfg.optim.warmup_epochs > 0
cfg.optim.warmup_init_lr = 0
cfg.optim.milestones = [40, 70]  # WarmupMultiStepLR
cfg.optim.use_gm_after_epoch = 20  # Sum loss will add permutation loss and verification loss
cfg.optim.warmup_multi_step_epochs = 120
cfg.optim.pretrain_new_params_epochs = 0  #
cfg.optim.pretrain_new_params = cfg.optim.pretrain_new_params_epochs > 0
cfg.optim.epochs_per_val = 10
cfg.optim.steps_per_log = 5
cfg.optim.trial_run = False  #
cfg.optim.phase = 'normal'  # [easy, pretrain, warmup, normal], may be re-configured in code
cfg.optim.resume = False
cfg.optim.resume_epoch = ''  #
cfg.optim.resume_from = 'pretrained'  # 'pretrained' or 'whole'
cfg.optim.cft = False  #
cfg.optim.cft_iters = 1  #
cfg.optim.cft_rho = 8e-4


# ==================================================MDRS====================================================== #
cfg.blocksetting = EasyDict()
cfg.blocksetting.backbone = 3
cfg.blocksetting.pool = 3
cfg.blocksetting.reduction = 8
cfg.blocksetting.classifier = 8
cfg.blocksetting.multi_seg = 3

cfg.vis = EasyDict()
cfg.vis.use = False
cfg.vis.heat_map = EasyDict()
cfg.vis.heat_map.use = False
cfg.vis.heat_map.save_dir = 'D:/weights_results/HJL-ReID/heat_map'

