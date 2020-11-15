HJL-re-id
=========
- An awesome project for person re-id. 
- This project contains adequate support for `log recording`, `loss monitoring` and `visualization ranked images`.
- This is the pytorch implementation the paper [*Joint multi-scale discrimination and region segmentation for person re-ID*](https://www.sciencedirect.com/science/article/pii/S0167865520303275#bib0023).

# Introduction
- This repository is for person re-id including supervised learning, unsupervised learning and occluded person re-id task. You can utilize this code to learn how to make `person re-id`tasks. 
- For HJL, this is my name initials -- Huang Jialiang.
- I has estabilished the structure containing the introduction of person re-id models like `PCB`,`MGN` and `MDRS`(my paper), `PGFA`, `Pyramid` and `HOReID`. These structures are all reproduced by myself in this code framework.
- If you have any quesions, please contact me by my email. 
- My emails is: *nickhuang1996@126.com*.

# Classical models 

These models are all reproduced by myself through this code framework.

- PCB [ECCV2018] [*Beyond Part Models: Person Retrieval with Refined Part Pooling (and A Strong Convolutional Baseline)*](https://arxiv.org/pdf/1711.09349.pdf)
- MGN [ACM Multimedia 2018] [*Learning Discriminative Features with Multiple Granularities for Person Re-Identification*](https://arxiv.org/abs/1804.01438)
- PGFA [ICCV2019] [*Pose-Guided Feature Alignment for Occluded Person Re-Identification*](https://ieeexplore.ieee.org/document/9010704)
- Pyramid [CVPR2019] [*Pyramidal Person Re-IDentification via Multi-Loss Dynamic Training*](https://openaccess.thecvf.com/content_CVPR_2019/papers/Zheng_Pyramidal_Person_Re-IDentification_via_Multi-Loss_Dynamic_Training_CVPR_2019_paper.pdf)
- HOReID [CVPR2020] [*High-Order Information Matters: Learning Relation and Topology for Occluded Person Re-Identification*](http://openaccess.thecvf.com/content_CVPR_2020/html/Wang_High-Order_Information_Matters_Learning_Relation_and_Topology_for_Occluded_Person_CVPR_2020_paper.html)

# Citation

If you find our work useful, please kindly cite our paper:
```
@article{huang2020MDRS,
  title={Joint multi-scale discrimination and region segmentation for person re-ID},
  author={Huang, Jialiang and Lio, bo and Fu, lihua},
  journal={Pattern Recognition Letters},
  year={2020}
}
```

# MDRS
- Name: [*Joint multi-scale discrimination and region segmentation for person re-ID*](https://www.sciencedirect.com/science/article/pii/S0167865520303275)
- Journal: [*Pattern Recognition Letters*](https://www.sciencedirect.com/journal/pattern-recognition-letters)
- [Volume 138](https://www.sciencedirect.com/journal/pattern-recognition-letters/vol/138/suppl/C), October 2020, Pages 540-547
- JCR: Q2
- web of science
## Architecture
![architecture.jpg](imgs/architecture.jpg)
 
 ## Ranked Images
![ranked_images.jpg](imgs/ranked_images.jpg)
- Re-ID examples of MDRS on Market-1501 dataset. The retrieved images are all from the gallery set with similarity scores shown above each image. Images with similarity scores in red are negative results. The introduction of multi-scale discriminative feature extraction and region segmentation boost the Re-ID performance.

### Dependencies
 - Python >= 3.6
 - Pytorch >= 1.0.0
 - Numpy
 - tqdm
 - scipy
 - torchvision==0.2.1
 
# Dataset Structure
- Main structure illustrates the data structure for person re-id.

### BaiduNetDisk Cloud
- Download link: [ReID_dataset](https://pan.baidu.com/s/1KyKi_3AROYF0nsZcFMISEA)
- Pwd: eseu

### Google Drive
- Download link: [ReID_dataset](https://drive.google.com/drive/folders/1xjdDK1Q7bbFwfFma9YSIxXS_cdh3jcAC)
### Notice
- `bounding_box_train_camstyle` are supposed to be in right place for unsupervised learning.
## Main Structure
- [market1501-structure](#market1501-structure)
- [duke-structure](#duke-structure)
- [cuhk03_np_detected_jpg-structure](#cuhk03_np_detected_jpg-structure)
```
${project_dir}/ReID_dataset
    market1501
        Market-1501-v15.09.15                   # Extracted from Market-1501-v15.09.15.zip, http://www.liangzheng.org/Project/project_reid.html
        Market-1501-v15.09.15_ps_label
        bounding_box_train_duke_style
    duke
        bounding_box_train_market1501_style
        DukeMTMC-reID                           # Extracted from DukeMTMC-reID.zip, https://github.com/layumi/DukeMTMC-reID_evaluation
        DukeMTMC-reID_ps_label
        Occluded_Duke
    cuhk03_np_detected_jpg
        cuhk03-np                               # Extracted from cuhk03-np.zip, https://pan.baidu.com/s/1RNvebTccjmmj1ig-LVjw7A
        cuhk03-np-jpg                           # Created from code
        cuhk03-np-jpg_ps_label
``` 
### market1501 structure
```
${project_dir}/ReID_dataset/market1501
    Market-1501-v15.09.15                       # Extracted from Market-1501-v15.09.15.zip, http://www.liangzheng.org/Project/project_reid.html
        bounding_box_test
        bounding_box_train
        bounding_box_train_camstyle             # unsupervised learning
        bounding_box_train_resize               # 256×256
        gt_bbox
        gt_query
        pytorch
            gallery
            multi-query
            query
            train
            train_all
            val
        query
    Market-1501-v15.09.15_ps_label              # segmentation label 
        bounding_box_test
        bounding_box_train
        gt_bbox
        query
    bounding_box_train_duke_style
``` 

### duke structure
```
${project_dir}/ReID_dataset/duke
    bounding_box_train_market1501_style
    DukeMTMC-reID                               # Extracted from DukeMTMC-reID.zip, https://github.com/layumi/DukeMTMC-reID_evaluation
        bounding_box_test
        bounding_box_train
        bounding_box_train_camstyle             # unsupervised learning
        pytorch
            gallery
            query
            train
            train_all
            val
        query
    DukeMTMC-reID_ps_label                      # segmentation label 
        bounding_box_test
        bounding_box_train
        gt_bbox
        query
    Occluded_Duke
        bounding_box_test
        bounding_box_train
        heatmaps                                # .npy files
            bounding_box_test
            bounding_box_train
            processed_data
                gallery
                query
                train 
        processed_data                          # person ID is folder name
            gallery
            query
            train
        query
        test_pose_storage
            gallery
                sep-json
                alphapose-results.json
                gallery.list
                show.py
            query
                sep-json
                1.list
                0184_c4_f0056533.jpg
                0184_c4_f0056533a.jpg
                alphapose-results.json
                show.py
                show1.py
``` 

### cuhk03_np_detected_jpg structure
```
${project_dir}/ReID_dataset/cuhk03_np_detected_jpg
    cuhk03-np                              # Extracted from cuhk03-np.zip, https://pan.baidu.com/s/1RNvebTccjmmj1ig-LVjw7A
        datected
            bounding_box_test
            bounding_box_train
            pytorch
                gallery
                query
                train
                train_all
                val
            query
        labeled
            bounding_box_test
            bounding_box_train
            query
    cuhk03-np-jpg                         # Created from code
        detected
            bounding_box_test
            bounding_box_train
            pytorch
                gallery
                query
                train
                train_all
                val
            query
        labeled
            bounding_box_test
            bounding_box_train
            pytorch
                gallery
                query
                train
                train_all
                val
            query
    cuhk03-np-jpg_ps_label
        bounding_box_test
        bounding_box_train
        query    
``` 

## Models
### BaiduNetDisk
- Download link: [weights_results](https://pan.baidu.com/s/1E3Iy1mAdui_VY7ydNB-RDw) 
- Pwd: ei5o
### Google Drive
- Download link: [weights_results](https://drive.google.com/drive/folders/1lY61BPeI8QAlKjrgikWSHMxs8eBTmG3z)
- Pre-trained Pose Model [pose_hrnet_w48_256x192.pth](https://drive.google.com/drive/folders/1hOTihvbyIxsm5ygDpbUuJ7O_tzv4oXjC) 
is supposed to be set into path ```${project_dir}/weights_results/HOReID/pre-trained/pose_model/pose_hrnet_w48_256x192.pth```
so that run `HOReID` task.

| Name | 
|------|
| ResNet50 |
| MDRS_ADAM_random_erasing_margin_0.3_market_best |
| MDRS_ADAM_random_erasing_margin_0.3_duke_best |
| MDRS_ADAM_random_erasing_margin_0.3_cuhk_jpg_best |

- Pretrained models(*ResNet50*) restores `backbone` weights.
- `HJL-ReID` is the project name, where three datasets results and weights are produced after training.
- `MDRS_feat_dict.mat` restores gallery and query data features by extracted from trained models for evaluations.
- You are supposed to download and save these files according to the following structure:
- `tensorboard` and `default_config.py` will be automatically created if you run codes. 
```
${project_dir}/weights_results
    pretrained_models
        resnet50-19c8e357.pth
    HJL-ReID
        MDRS_ADAM_random_erasing_margin_0.3_market_best
            tensorboard
            ckpt.pth
            default_config.py
            MDRS_feat_dict.mat
        MDRS_ADAM_random_erasing_margin_0.3_duke_best
            ...
        MDRS_ADAM_random_erasing_margin_0.3_cuhk_jpg_best
            ...
        ...
```

# Train and Test
### No IDE
- if you want to command, followed by:
- `python demo.py --model_name MDRS
    --exp_dir ${project_dir}/weights_results/HJL-ReID/MDRS_ADAM_random_erasing_margin_0.3_market_best
    --default_config_path $(project_dir)/HJL-ReID/MDRSREID/Settings/config/default_config.py
    --ow_config_path $(project_dir)/HJL-ReID/MDRSREID/Settings/config/overwrite_config/MDRS_config_ADAM_best_market1501.txt
    --ow_str "cfg.dataset.train.name = 'market1501'"`
### IDE
- or if you have IDE like `pycharm`, you can just modify `MDRSREID/parser_args/parser_args.py`
- You just modify `--exp_dir` and `--ow_config_path`. Then run `demo.py`.

|dataset| exp_dir | ow_config_path |
|---|---|---|
| market | ${project_dir}/weights_results/HJL-ReID/MDRS_ADAM_random_erasing_margin_0.3_market_best | $(project_dir)/HJL-ReID/MDRSREID/Settings/config/overwrite_config/MDRS_config_ADAM_best_market1501.txt |
| duke | ${project_dir}/weights_results/HJL-ReID/MDRS_ADAM_random_erasing_margin_0.3_duke_best | $(project_dir)/HJL-ReID/MDRSREID/Settings/config/overwrite_config/MDRS_config_ADAM_best_duke.txt |
| cuhk03 | ${project_dir}/weights_results/HJL-ReID/MDRS_ADAM_random_erasing_margin_0.3_cuhk_jpg_best | $(project_dir)/HJL-ReID/MDRSREID/Settings/config/overwrite_config/MDRS_config_ADAM_best_cuhk03_jpg.txt |
- If you want to test, just modify `cfg.only_test = False` to `cfg.only_test = True` in `./MDRSREID/Settings/config/overwrite_config/${config_file}`

### Config files
Overwrite config files can be found in: [./MDRSREID/Settings/config/overwrite_config/](MDRSREID/Settings/config/overwrite_config)

| config_file |
| -------------- |
| MDRS_config_ADAM_best_market1501.txt |
| MDRS_config_ADAM_best_duke.txt |
| MDRS_config_ADAM_best_cuhk_jpg.txt |
| ... |
- e.g:
```
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
                        default='D:/weights_results/HJL-ReID/MDRS_ADAM_random_erasing_margin_0.3_market_best',
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

```
# Evaluation

- For saving time on feature extraction, I have provided three `MDRS_feat_dict.mat` in `weights_results`. You can use these `feat_dict.mat` to test models.
- Download link：[weights_results](https://drive.google.com/drive/folders/1lY61BPeI8QAlKjrgikWSHMxs8eBTmG3z)

### Warning
- These `feat_dict.mat` files are for testing so please avoid them being rewritten if you set `cfg.only_test = False`.

The stdout files as below：
### Market1501
```
=> Experiment Output Directory: D:/weights_results/HJL-ReID/MDRS_ADAM_random_erasing_margin_0.3_market_best
[PYTORCH VERSION]: 1.0.1
 keypoints_model :
	 test :
		 model_file : D:/weights_results/HOReID/pre-trained/pose_model/pose_hrnet_w48_256x192.pth
	 is_train : False
	 name : pose_hrnet
	 norm_scale : 10.0
	 num_joints : 17
	 extra :
		 pretrained_layers : ['*']
		 stem_inplanes : 64
		 stage2 :
			 num_modules : 1
			 num_branches : 2
			 num_blocks : [4, 4]
			 num_channels : [48, 96]
			 block : BASIC
			 fuse_method : SUM
		 stage3 :
			 num_modules : 4
			 num_branches : 3
			 num_blocks : [4, 4, 4]
			 num_channels : [48, 96, 192]
			 block : BASIC
			 fuse_method : SUM
		 stage4 :
			 num_modules : 3
			 num_branches : 4
			 num_blocks : [4, 4, 4, 4]
			 num_channels : [48, 96, 192, 384]
			 block : BASIC
			 fuse_method : SUM
		 final_conv_kernel : 1
	 init_weights : True
	 pretrained : 
	 joints_groups : [[0, 1, 2, 3, 4], [5], [6], [7], [8], [9], [10], [11], [12], [13], [14], [15], [16]]
	 branch_num : 14
	 weight_global_feature : 1.0
 only_test : True
 only_infer : False
 model :
	 name : MDRS
	 backbone :
		 name : resnet50
		 last_conv_stride : 2
		 pretrained : True
		 pretrained_model_dir : D:/weights_results/pretrained_model/imagenet_model
	 pool_type : MDRSPool
	 max_or_avg : max
	 em_dim : 256
	 num_parts : 1
	 use_ps : True
	 reduction :
		 use_relu : False
		 use_leakyrelu : False
		 use_dropout : False
	 PGFA :
		 global_input_dim : 4096
		 part_feat_input_dim : 2048
	 multi_seg :
		 mid_c : 256
		 mid_c2 : 512
		 in_c1 : 2048
		 in_c2 : 1024
		 num_classes : 8
	 gcn :
		 scale : 20.0
	 num_classes : 751
 dataset :
	 root : D:/datasets/ReID_dataset
	 use_occlude_duke : False
	 im :
		 h_w : [384, 128]
		 mean : [0.486, 0.459, 0.408]
		 std : [0.229, 0.224, 0.225]
		 interpolation : None
		 pad : 10
		 random_erasing :
			 epsilon : 0.5
			 proportion : True
		 random_crop :
			 output_size : [256, 128]
	 use_pose_landmark_mask : False
	 pose_landmark_mask :
		 h_w : [24, 8]
		 type : PL_18P
	 use_ps_label : True
	 ps_label :
		 h_w : [48, 16]
		 pad : 1
	 train :
		 before_to_tensor_transform_list : ['hflip', 'resize']
		 after_to_tensor_transform_list : ['random_erasing']
		 type : Supervised
		 source :
			 name : market1501
			 authority : train
		 target :
			 name : market1501
			 authority : train
	 cd_train :
		 name : duke
		 authority : train
		 transform_list : ['hflip', 'resize']
	 test :
		 names : ['market1501']
		 before_to_tensor_transform_list : ['resize']
		 after_to_tensor_transform_list : None
 model_flow : train
 stage : FeatureExtract
 dataloader :
	 num_workers : 2
	 train :
		 batch_type : pk2
		 batch_size : 32
		 batch_id : 8
		 batch_image : 4
		 drop_last : True
	 cd_train :
		 batch_type : random
		 batch_size : 32
		 drop_last : True
	 test :
		 batch_type : seq
		 batch_size : 16
		 drop_last : False
	 pk :
		 k : 4
	 pk2 :
 eval :
	 forward_type : reid
	 chunk_size : 1000
	 re_rank : False
	 separate_camera_set : False
	 single_gallery_shot : False
	 first_match_break : True
	 score_prefix : 
	 ranked_images : False
	 ver_in_scale : 10.0
	 feat_dict :
		 use : True
		 reproduction : False
	 device : cuda
 train :
 id_loss :
	 name : idL
	 weight : 1
	 use : True
 id_smooth_loss :
	 name : idSmoothL
	 weight : 0
	 use : False
	 epsilon : 0.1
	 reduce : True
	 use_gpu : True
 tri_loss :
	 name : triL
	 weight : 2
	 use : True
	 margin : 0.3
	 dist_type : euclidean
	 hard_type : tri_hard
	 norm_by_num_of_effective_triplets : False
 tri_hard_loss :
	 name : triHardL
	 weight : 0
	 use : False
	 margin : 0.3
	 dist_type : euclidean
 permutation_loss :
	 name : permutationL
	 weight : 0
	 use : False
	 branch_num : 14
 verification_loss :
	 name : verL
	 weight : 0
	 use : False
 src_multi_seg_loss :
	 name : psL
	 weight : 1
	 use : False
	 normalize_size : True
	 num_classes : 8
 src_multi_seg_gp_loss :
	 name : psL
	 weight : 0
	 use : False
	 normalize_size : True
	 num_classes : 8
 pgfa_loss :
	 name : pgfaL
	 weight : 0
	 use : False
	 lamb : 0.2
 inv_loss :
	 name : InvL
	 num_features : 2048
	 weight : 0
	 use : False
	 beta : 0.05
	 alpha : 0.01
	 knn : 6
	 lmd : 0.3
 analyze_computer :
	 name : analyze_computer
	 use : False
 verification_probability_analyze :
	 name : verification_probability_analyze
	 use : False
 log :
	 use_tensorboard : True
	 only_base_lr : False
	 exp_dir : D:/weights_results/HJL-ReID/MDRS_ADAM_random_erasing_margin_0.3_market_best
	 ckpt_file : D:/weights_results/HJL-ReID/MDRS_ADAM_random_erasing_margin_0.3_market_best\ckpt.pth
	 score_file : D:/weights_results/HJL-ReID/MDRS_ADAM_random_erasing_margin_0.3_market_best\score_2020-09-19_14-16-16.txt
 optim :
	 optimizer : adam
	 sgd :
		 momentum : 0.9
		 nesterov : False
		 weight_decay : 0.0005
	 adam :
		 beta1 : 0.9
		 beta2 : 0.99
		 eps : 1e-08
		 amsgrad : False
		 weight_decay : 0.0005
	 seperate_params : False
	 base_lr : 0.00035
	 ft_lr : 0.0002
	 new_params_lr : 0.0002
	 gcn_lr_scale : 0.1
	 gm_lr_scale : 1.0
	 ver_lr_scale : 1.0
	 every_lr_epoch : 40
	 easy_epochs : 60
	 lr_decay_epochs : [320, 360, 400]
	 normal_epochs : 400
	 warmup_epochs : 0
	 warmup : False
	 warmup_init_lr : 0
	 milestones : [40, 70]
	 use_gm_after_epoch : 20
	 warmup_multi_step_epochs : 120
	 pretrain_new_params_epochs : 0
	 pretrain_new_params : False
	 epochs_per_val : 20
	 steps_per_log : 5
	 trial_run : False
	 phase : normal
	 resume : False
	 resume_epoch : 
	 resume_from : pretrained
	 cft : False
	 cft_iters : 1
	 cft_rho : 0.0008
 blocksetting :
	 backbone : 3
	 pool : 3
	 reduction : 8
	 classifier : 8
	 multi_seg : 3
 vis :
	 use : False
	 heat_map :
		 use : False
		 save_dir : D:/weights_results/HJL-ReID/heat_map
 device : cuda
Keys not found in source state_dict: 
	 num_batches_tracked  x53
Keys not found in destination state_dict: 
	 fc.weight
	 fc.bias
=> Loaded ImageNet Model: D:/weights_results/pretrained_model/imagenet_model\resnet50-19c8e357.pth
Model(
  (model): MDRS(
    (backbone): MDRSBackbone(
      (backbone): Sequential(
        (0): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace)
        (3): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        (4): Sequential(
          (0): Bottleneck(
            (conv1): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace)
            (downsample): Sequential(
              (0): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
          (1): Bottleneck(
            (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace)
          )
          (2): Bottleneck(
            (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace)
          )
        )
        (5): Sequential(
          (0): Bottleneck(
            (conv1): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace)
            (downsample): Sequential(
              (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
              (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
          (1): Bottleneck(
            (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace)
          )
          (2): Bottleneck(
            (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace)
          )
          (3): Bottleneck(
            (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace)
          )
        )
        (6): Bottleneck(
          (conv1): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace)
          (downsample): Sequential(
            (0): Conv2d(512, 1024, kernel_size=(1, 1), stride=(2, 2), bias=False)
            (1): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
      )
      (p0): Sequential(
        (0): Sequential(
          (0): Bottleneck(
            (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace)
          )
          (1): Bottleneck(
            (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace)
          )
          (2): Bottleneck(
            (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace)
          )
          (3): Bottleneck(
            (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace)
          )
          (4): Bottleneck(
            (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace)
          )
        )
        (1): Sequential(
          (0): Bottleneck(
            (conv1): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace)
            (downsample): Sequential(
              (0): Conv2d(1024, 2048, kernel_size=(1, 1), stride=(2, 2), bias=False)
              (1): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
          (1): Bottleneck(
            (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace)
          )
          (2): Bottleneck(
            (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace)
          )
        )
      )
      (p1): Sequential(
        (0): Sequential(
          (0): Bottleneck(
            (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace)
          )
          (1): Bottleneck(
            (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace)
          )
          (2): Bottleneck(
            (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace)
          )
          (3): Bottleneck(
            (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace)
          )
          (4): Bottleneck(
            (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace)
          )
        )
        (1): Sequential(
          (0): Bottleneck(
            (conv1): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace)
            (downsample): Sequential(
              (0): Conv2d(1024, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
          (1): Bottleneck(
            (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace)
          )
          (2): Bottleneck(
            (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace)
          )
        )
      )
      (p2): Sequential(
        (0): Sequential(
          (0): Bottleneck(
            (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace)
          )
          (1): Bottleneck(
            (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace)
          )
          (2): Bottleneck(
            (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace)
          )
          (3): Bottleneck(
            (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace)
          )
          (4): Bottleneck(
            (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace)
          )
        )
        (1): Sequential(
          (0): Bottleneck(
            (conv1): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace)
            (downsample): Sequential(
              (0): Conv2d(1024, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
          (1): Bottleneck(
            (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace)
          )
          (2): Bottleneck(
            (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace)
          )
        )
      )
    )
    (pool): MDRSPool(
      (pool0): AdaptiveMaxPool2d(output_size=(1, 1))
      (pool1): AdaptiveMaxPool2d(output_size=(2, 1))
      (pool2): AdaptiveMaxPool2d(output_size=(3, 1))
    )
    (reduction): MDRSReduction(
      (reduction): ModuleList(
        (0): Sequential(
          (0): Conv2d(2048, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU()
        )
        (1): Sequential(
          (0): Conv2d(2048, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU()
        )
        (2): Sequential(
          (0): Conv2d(2048, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU()
        )
        (3): Sequential(
          (0): Conv2d(2048, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU()
        )
        (4): Sequential(
          (0): Conv2d(2048, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU()
        )
        (5): Sequential(
          (0): Conv2d(2048, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU()
        )
        (6): Sequential(
          (0): Conv2d(2048, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU()
        )
        (7): Sequential(
          (0): Conv2d(2048, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU()
        )
      )
    )
    (classifier): MDRSClassifier(
      (classifier): ModuleList(
        (0): Linear(in_features=256, out_features=751, bias=True)
        (1): Linear(in_features=256, out_features=751, bias=True)
        (2): Linear(in_features=256, out_features=751, bias=True)
        (3): Linear(in_features=256, out_features=751, bias=True)
        (4): Linear(in_features=256, out_features=751, bias=True)
        (5): Linear(in_features=256, out_features=751, bias=True)
        (6): Linear(in_features=256, out_features=751, bias=True)
        (7): Linear(in_features=256, out_features=751, bias=True)
      )
    )
    (multi_seg): MDRSMultiSeg(
      (deconv): ConvTranspose2d(2048, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1), bias=False)
      (deconv3): ConvTranspose2d(2048, 1024, kernel_size=(3, 3), stride=(3, 3), padding=(1, 1), output_padding=(2, 2), bias=False)
      (deconv3_pool): AdaptiveAvgPool2d(output_size=(24, 8))
      (deconv1): ConvTranspose2d(2048, 1024, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1), bias=False)
      (deconv2): ConvTranspose2d(1024, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1), bias=False)
      (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace)
      (conv): Conv2d(256, 8, kernel_size=(1, 1), stride=(1, 1))
    )
  )
)
Model size: 113.637 M
=> Loaded [model] from D:/weights_results/HJL-ReID/MDRS_ADAM_random_erasing_margin_0.3_market_best\ckpt.pth, epoch 440, score:
market1501 -> market1501      [mAP:  87.5%], [cmc1:  95.1%], [cmc5:  98.2%], [cmc10:  98.8%]
check query dataset success!!
check gallery dataset success!!
Test...
Loading feat dict mat from D:/weights_results/HJL-ReID/MDRS_ADAM_random_erasing_margin_0.3_market_best/MDRS_feat_dict.mat...
D:/weights_results/HJL-ReID/MDRS_ADAM_random_erasing_margin_0.3_market_best/MDRS_feat_dict.mat has been existed.
market1501 -> market1501      [mAP:  87.5%], [cmc1:  95.1%], [cmc5:  98.2%], [cmc10:  98.8%]
```

### DukeMTMC-reID
```
=> Experiment Output Directory: D:/weights_results/HJL-ReID/MDRS_ADAM_random_erasing_margin_0.3_duke_best
[PYTORCH VERSION]: 1.0.1
 keypoints_model :
	 test :
		 model_file : D:/weights_results/HOReID/pre-trained/pose_model/pose_hrnet_w48_256x192.pth
	 is_train : False
	 name : pose_hrnet
	 norm_scale : 10.0
	 num_joints : 17
	 extra :
		 pretrained_layers : ['*']
		 stem_inplanes : 64
		 stage2 :
			 num_modules : 1
			 num_branches : 2
			 num_blocks : [4, 4]
			 num_channels : [48, 96]
			 block : BASIC
			 fuse_method : SUM
		 stage3 :
			 num_modules : 4
			 num_branches : 3
			 num_blocks : [4, 4, 4]
			 num_channels : [48, 96, 192]
			 block : BASIC
			 fuse_method : SUM
		 stage4 :
			 num_modules : 3
			 num_branches : 4
			 num_blocks : [4, 4, 4, 4]
			 num_channels : [48, 96, 192, 384]
			 block : BASIC
			 fuse_method : SUM
		 final_conv_kernel : 1
	 init_weights : True
	 pretrained : 
	 joints_groups : [[0, 1, 2, 3, 4], [5], [6], [7], [8], [9], [10], [11], [12], [13], [14], [15], [16]]
	 branch_num : 14
	 weight_global_feature : 1.0
 only_test : True
 only_infer : False
 model :
	 name : MDRS
	 backbone :
		 name : resnet50
		 last_conv_stride : 2
		 pretrained : True
		 pretrained_model_dir : D:/weights_results/pretrained_model/imagenet_model
	 pool_type : MDRSPool
	 max_or_avg : max
	 em_dim : 256
	 num_parts : 1
	 use_ps : True
	 reduction :
		 use_relu : False
		 use_leakyrelu : False
		 use_dropout : False
	 PGFA :
		 global_input_dim : 4096
		 part_feat_input_dim : 2048
	 multi_seg :
		 mid_c : 256
		 mid_c2 : 512
		 in_c1 : 2048
		 in_c2 : 1024
		 num_classes : 8
	 gcn :
		 scale : 20.0
	 num_classes : 702
 dataset :
	 root : D:/datasets/ReID_dataset
	 use_occlude_duke : False
	 im :
		 h_w : [384, 128]
		 mean : [0.486, 0.459, 0.408]
		 std : [0.229, 0.224, 0.225]
		 interpolation : None
		 pad : 10
		 random_erasing :
			 epsilon : 0.5
			 proportion : True
		 random_crop :
			 output_size : [256, 128]
	 use_pose_landmark_mask : False
	 pose_landmark_mask :
		 h_w : [24, 8]
		 type : PL_18P
	 use_ps_label : True
	 ps_label :
		 h_w : [48, 16]
		 pad : 1
	 train :
		 before_to_tensor_transform_list : ['hflip', 'resize']
		 after_to_tensor_transform_list : ['random_erasing']
		 type : Supervised
		 source :
			 name : duke
			 authority : train
		 target :
			 name : duke
			 authority : train
	 cd_train :
		 name : duke
		 authority : train
		 transform_list : ['hflip', 'resize']
	 test :
		 names : ['duke']
		 before_to_tensor_transform_list : ['resize']
		 after_to_tensor_transform_list : None
 model_flow : train
 stage : FeatureExtract
 dataloader :
	 num_workers : 2
	 train :
		 batch_type : pk2
		 batch_size : 32
		 batch_id : 8
		 batch_image : 4
		 drop_last : True
	 cd_train :
		 batch_type : random
		 batch_size : 32
		 drop_last : True
	 test :
		 batch_type : seq
		 batch_size : 16
		 drop_last : False
	 pk :
		 k : 4
	 pk2 :
 eval :
	 forward_type : reid
	 chunk_size : 1000
	 re_rank : False
	 separate_camera_set : False
	 single_gallery_shot : False
	 first_match_break : True
	 score_prefix : 
	 ranked_images : False
	 ver_in_scale : 10.0
	 feat_dict :
		 use : True
		 reproduction : False
	 device : cuda
 train :
 id_loss :
	 name : idL
	 weight : 1
	 use : True
 id_smooth_loss :
	 name : idSmoothL
	 weight : 0
	 use : False
	 epsilon : 0.1
	 reduce : True
	 use_gpu : True
 tri_loss :
	 name : triL
	 weight : 2
	 use : True
	 margin : 0.3
	 dist_type : euclidean
	 hard_type : tri_hard
	 norm_by_num_of_effective_triplets : False
 tri_hard_loss :
	 name : triHardL
	 weight : 0
	 use : False
	 margin : 0.3
	 dist_type : euclidean
 permutation_loss :
	 name : permutationL
	 weight : 0
	 use : False
	 branch_num : 14
 verification_loss :
	 name : verL
	 weight : 0
	 use : False
 src_multi_seg_loss :
	 name : psL
	 weight : 1
	 use : False
	 normalize_size : True
	 num_classes : 8
 src_multi_seg_gp_loss :
	 name : psL
	 weight : 0
	 use : False
	 normalize_size : True
	 num_classes : 8
 pgfa_loss :
	 name : pgfaL
	 weight : 0
	 use : False
	 lamb : 0.2
 inv_loss :
	 name : InvL
	 num_features : 2048
	 weight : 0
	 use : False
	 beta : 0.05
	 alpha : 0.01
	 knn : 6
	 lmd : 0.3
 analyze_computer :
	 name : analyze_computer
	 use : False
 verification_probability_analyze :
	 name : verification_probability_analyze
	 use : False
 log :
	 use_tensorboard : True
	 only_base_lr : False
	 exp_dir : D:/weights_results/HJL-ReID/MDRS_ADAM_random_erasing_margin_0.3_duke_best
	 ckpt_file : D:/weights_results/HJL-ReID/MDRS_ADAM_random_erasing_margin_0.3_duke_best\ckpt.pth
	 score_file : D:/weights_results/HJL-ReID/MDRS_ADAM_random_erasing_margin_0.3_duke_best\score_2020-09-19_13-26-20.txt
 optim :
	 optimizer : adam
	 sgd :
		 momentum : 0.9
		 nesterov : False
		 weight_decay : 0.0005
	 adam :
		 beta1 : 0.9
		 beta2 : 0.99
		 eps : 1e-08
		 amsgrad : False
		 weight_decay : 0.0005
	 seperate_params : False
	 base_lr : 0.00035
	 ft_lr : 0.0002
	 new_params_lr : 0.0002
	 gcn_lr_scale : 0.1
	 gm_lr_scale : 1.0
	 ver_lr_scale : 1.0
	 every_lr_epoch : 40
	 easy_epochs : 60
	 lr_decay_epochs : [500, 560, 600]
	 normal_epochs : 600
	 warmup_epochs : 0
	 warmup : False
	 warmup_init_lr : 0
	 milestones : [40, 70]
	 use_gm_after_epoch : 20
	 warmup_multi_step_epochs : 120
	 pretrain_new_params_epochs : 0
	 pretrain_new_params : False
	 epochs_per_val : 20
	 steps_per_log : 5
	 trial_run : False
	 phase : normal
	 resume : False
	 resume_epoch : 
	 resume_from : pretrained
	 cft : False
	 cft_iters : 1
	 cft_rho : 0.0008
 blocksetting :
	 backbone : 3
	 pool : 3
	 reduction : 8
	 classifier : 8
	 multi_seg : 3
 vis :
	 use : False
	 heat_map :
		 use : False
		 save_dir : D:/weights_results/HJL-ReID/heat_map
 device : cuda
Keys not found in source state_dict: 
	 num_batches_tracked  x53
Keys not found in destination state_dict: 
	 fc.bias
	 fc.weight
=> Loaded ImageNet Model: D:/weights_results/pretrained_model/imagenet_model\resnet50-19c8e357.pth
Model(
  (model): MDRS(
    (backbone): MDRSBackbone(
      (backbone): Sequential(
        (0): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace)
        (3): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        (4): Sequential(
          (0): Bottleneck(
            (conv1): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace)
            (downsample): Sequential(
              (0): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
          (1): Bottleneck(
            (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace)
          )
          (2): Bottleneck(
            (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace)
          )
        )
        (5): Sequential(
          (0): Bottleneck(
            (conv1): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace)
            (downsample): Sequential(
              (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
              (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
          (1): Bottleneck(
            (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace)
          )
          (2): Bottleneck(
            (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace)
          )
          (3): Bottleneck(
            (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace)
          )
        )
        (6): Bottleneck(
          (conv1): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace)
          (downsample): Sequential(
            (0): Conv2d(512, 1024, kernel_size=(1, 1), stride=(2, 2), bias=False)
            (1): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
      )
      (p0): Sequential(
        (0): Sequential(
          (0): Bottleneck(
            (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace)
          )
          (1): Bottleneck(
            (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace)
          )
          (2): Bottleneck(
            (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace)
          )
          (3): Bottleneck(
            (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace)
          )
          (4): Bottleneck(
            (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace)
          )
        )
        (1): Sequential(
          (0): Bottleneck(
            (conv1): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace)
            (downsample): Sequential(
              (0): Conv2d(1024, 2048, kernel_size=(1, 1), stride=(2, 2), bias=False)
              (1): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
          (1): Bottleneck(
            (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace)
          )
          (2): Bottleneck(
            (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace)
          )
        )
      )
      (p1): Sequential(
        (0): Sequential(
          (0): Bottleneck(
            (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace)
          )
          (1): Bottleneck(
            (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace)
          )
          (2): Bottleneck(
            (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace)
          )
          (3): Bottleneck(
            (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace)
          )
          (4): Bottleneck(
            (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace)
          )
        )
        (1): Sequential(
          (0): Bottleneck(
            (conv1): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace)
            (downsample): Sequential(
              (0): Conv2d(1024, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
          (1): Bottleneck(
            (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace)
          )
          (2): Bottleneck(
            (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace)
          )
        )
      )
      (p2): Sequential(
        (0): Sequential(
          (0): Bottleneck(
            (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace)
          )
          (1): Bottleneck(
            (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace)
          )
          (2): Bottleneck(
            (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace)
          )
          (3): Bottleneck(
            (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace)
          )
          (4): Bottleneck(
            (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace)
          )
        )
        (1): Sequential(
          (0): Bottleneck(
            (conv1): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace)
            (downsample): Sequential(
              (0): Conv2d(1024, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
          (1): Bottleneck(
            (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace)
          )
          (2): Bottleneck(
            (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace)
          )
        )
      )
    )
    (pool): MDRSPool(
      (pool0): AdaptiveMaxPool2d(output_size=(1, 1))
      (pool1): AdaptiveMaxPool2d(output_size=(2, 1))
      (pool2): AdaptiveMaxPool2d(output_size=(3, 1))
    )
    (reduction): MDRSReduction(
      (reduction): ModuleList(
        (0): Sequential(
          (0): Conv2d(2048, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU()
        )
        (1): Sequential(
          (0): Conv2d(2048, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU()
        )
        (2): Sequential(
          (0): Conv2d(2048, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU()
        )
        (3): Sequential(
          (0): Conv2d(2048, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU()
        )
        (4): Sequential(
          (0): Conv2d(2048, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU()
        )
        (5): Sequential(
          (0): Conv2d(2048, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU()
        )
        (6): Sequential(
          (0): Conv2d(2048, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU()
        )
        (7): Sequential(
          (0): Conv2d(2048, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU()
        )
      )
    )
    (classifier): MDRSClassifier(
      (classifier): ModuleList(
        (0): Linear(in_features=256, out_features=702, bias=True)
        (1): Linear(in_features=256, out_features=702, bias=True)
        (2): Linear(in_features=256, out_features=702, bias=True)
        (3): Linear(in_features=256, out_features=702, bias=True)
        (4): Linear(in_features=256, out_features=702, bias=True)
        (5): Linear(in_features=256, out_features=702, bias=True)
        (6): Linear(in_features=256, out_features=702, bias=True)
        (7): Linear(in_features=256, out_features=702, bias=True)
      )
    )
    (multi_seg): MDRSMultiSeg(
      (deconv): ConvTranspose2d(2048, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1), bias=False)
      (deconv3): ConvTranspose2d(2048, 1024, kernel_size=(3, 3), stride=(3, 3), padding=(1, 1), output_padding=(2, 2), bias=False)
      (deconv3_pool): AdaptiveAvgPool2d(output_size=(24, 8))
      (deconv1): ConvTranspose2d(2048, 1024, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1), bias=False)
      (deconv2): ConvTranspose2d(1024, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1), bias=False)
      (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace)
      (conv): Conv2d(256, 8, kernel_size=(1, 1), stride=(1, 1))
    )
  )
)
Model size: 113.637 M
=> Loaded [model] from D:/weights_results/HJL-ReID/MDRS_ADAM_random_erasing_margin_0.3_duke_best\ckpt.pth, epoch 600, score:
duke -> duke                  [mAP:  79.8%], [cmc1:  89.4%], [cmc5:  94.9%], [cmc10:  96.1%]
Test...
Loading feat dict mat from D:/weights_results/HJL-ReID/MDRS_ADAM_random_erasing_margin_0.3_duke_best/MDRS_feat_dict.mat...
D:/weights_results/HJL-ReID/MDRS_ADAM_random_erasing_margin_0.3_duke_best/MDRS_feat_dict.mat has been existed.
duke -> duke                  [mAP:  79.8%], [cmc1:  89.4%], [cmc5:  94.9%], [cmc10:  96.1%]
destination path dir D:/weights_results/HJL-ReID/MDRS_ADAM_random_erasing_margin_0.3_duke_best has been already exist.
=> Checkpoint Saved to D:/weights_results/HJL-ReID/MDRS_ADAM_random_erasing_margin_0.3_duke_best\ckpt.pth
```

### CUHK03
```
=> Experiment Output Directory: D:/weights_results/HJL-ReID/MDRS_ADAM_random_erasing_margin_0.3_cuhk_jpg_best
[PYTORCH VERSION]: 1.0.1
 keypoints_model :
	 test :
		 model_file : D:/weights_results/HOReID/pre-trained/pose_model/pose_hrnet_w48_256x192.pth
	 is_train : False
	 name : pose_hrnet
	 norm_scale : 10.0
	 num_joints : 17
	 extra :
		 pretrained_layers : ['*']
		 stem_inplanes : 64
		 stage2 :
			 num_modules : 1
			 num_branches : 2
			 num_blocks : [4, 4]
			 num_channels : [48, 96]
			 block : BASIC
			 fuse_method : SUM
		 stage3 :
			 num_modules : 4
			 num_branches : 3
			 num_blocks : [4, 4, 4]
			 num_channels : [48, 96, 192]
			 block : BASIC
			 fuse_method : SUM
		 stage4 :
			 num_modules : 3
			 num_branches : 4
			 num_blocks : [4, 4, 4, 4]
			 num_channels : [48, 96, 192, 384]
			 block : BASIC
			 fuse_method : SUM
		 final_conv_kernel : 1
	 init_weights : True
	 pretrained : 
	 joints_groups : [[0, 1, 2, 3, 4], [5], [6], [7], [8], [9], [10], [11], [12], [13], [14], [15], [16]]
	 branch_num : 14
	 weight_global_feature : 1.0
 only_test : True
 only_infer : False
 model :
	 name : MDRS
	 backbone :
		 name : resnet50
		 last_conv_stride : 2
		 pretrained : True
		 pretrained_model_dir : D:/weights_results/pretrained_model/imagenet_model
	 pool_type : MDRSPool
	 max_or_avg : max
	 em_dim : 256
	 num_parts : 1
	 use_ps : True
	 reduction :
		 use_relu : False
		 use_leakyrelu : False
		 use_dropout : False
	 PGFA :
		 global_input_dim : 4096
		 part_feat_input_dim : 2048
	 multi_seg :
		 mid_c : 256
		 mid_c2 : 512
		 in_c1 : 2048
		 in_c2 : 1024
		 num_classes : 8
	 gcn :
		 scale : 20.0
	 num_classes : 0
 dataset :
	 root : D:/datasets/ReID_dataset
	 use_occlude_duke : False
	 im :
		 h_w : [384, 128]
		 mean : [0.486, 0.459, 0.408]
		 std : [0.229, 0.224, 0.225]
		 interpolation : None
		 pad : 10
		 random_erasing :
			 epsilon : 0.5
			 proportion : True
		 random_crop :
			 output_size : [256, 128]
	 use_pose_landmark_mask : False
	 pose_landmark_mask :
		 h_w : [24, 8]
		 type : PL_18P
	 use_ps_label : True
	 ps_label :
		 h_w : [48, 16]
		 pad : 1
	 train :
		 before_to_tensor_transform_list : ['hflip', 'resize']
		 after_to_tensor_transform_list : ['random_erasing']
		 type : Supervised
		 source :
			 name : cuhk03_np_detected_jpg
			 authority : train
		 target :
			 name : cuhk03_np_detected_jpg
			 authority : train
	 cd_train :
		 name : duke
		 authority : train
		 transform_list : ['hflip', 'resize']
	 test :
		 names : ['cuhk03_np_detected_jpg']
		 before_to_tensor_transform_list : ['resize']
		 after_to_tensor_transform_list : None
 model_flow : train
 stage : FeatureExtract
 dataloader :
	 num_workers : 2
	 train :
		 batch_type : pk2
		 batch_size : 32
		 batch_id : 8
		 batch_image : 4
		 drop_last : True
	 cd_train :
		 batch_type : random
		 batch_size : 32
		 drop_last : True
	 test :
		 batch_type : seq
		 batch_size : 16
		 drop_last : False
	 pk :
		 k : 4
	 pk2 :
 eval :
	 forward_type : reid
	 chunk_size : 1000
	 re_rank : False
	 separate_camera_set : False
	 single_gallery_shot : False
	 first_match_break : True
	 score_prefix : 
	 ranked_images : False
	 ver_in_scale : 10.0
	 feat_dict :
		 use : True
		 reproduction : False
	 device : cuda
 train :
 id_loss :
	 name : idL
	 weight : 1
	 use : True
 id_smooth_loss :
	 name : idSmoothL
	 weight : 0
	 use : False
	 epsilon : 0.1
	 reduce : True
	 use_gpu : True
 tri_loss :
	 name : triL
	 weight : 2
	 use : True
	 margin : 0.3
	 dist_type : euclidean
	 hard_type : tri_hard
	 norm_by_num_of_effective_triplets : False
 tri_hard_loss :
	 name : triHardL
	 weight : 0
	 use : False
	 margin : 0.3
	 dist_type : euclidean
 permutation_loss :
	 name : permutationL
	 weight : 0
	 use : False
	 branch_num : 14
 verification_loss :
	 name : verL
	 weight : 0
	 use : False
 src_multi_seg_loss :
	 name : psL
	 weight : 1
	 use : False
	 normalize_size : True
	 num_classes : 8
 src_multi_seg_gp_loss :
	 name : psL
	 weight : 0
	 use : False
	 normalize_size : True
	 num_classes : 8
 pgfa_loss :
	 name : pgfaL
	 weight : 0
	 use : False
	 lamb : 0.2
 inv_loss :
	 name : InvL
	 num_features : 2048
	 weight : 0
	 use : False
	 beta : 0.05
	 alpha : 0.01
	 knn : 6
	 lmd : 0.3
 analyze_computer :
	 name : analyze_computer
	 use : False
 verification_probability_analyze :
	 name : verification_probability_analyze
	 use : False
 log :
	 use_tensorboard : True
	 only_base_lr : False
	 exp_dir : D:/weights_results/HJL-ReID/MDRS_ADAM_random_erasing_margin_0.3_cuhk_jpg_best
	 ckpt_file : D:/weights_results/HJL-ReID/MDRS_ADAM_random_erasing_margin_0.3_cuhk_jpg_best\ckpt.pth
	 score_file : D:/weights_results/HJL-ReID/MDRS_ADAM_random_erasing_margin_0.3_cuhk_jpg_best\score_2020-09-20_16-45-54.txt
 optim :
	 optimizer : adam
	 sgd :
		 momentum : 0.9
		 nesterov : False
		 weight_decay : 0.0005
	 adam :
		 beta1 : 0.9
		 beta2 : 0.99
		 eps : 1e-08
		 amsgrad : False
		 weight_decay : 0.0005
	 seperate_params : False
	 base_lr : 0.00035
	 ft_lr : 0.0002
	 new_params_lr : 0.0002
	 gcn_lr_scale : 0.1
	 gm_lr_scale : 1.0
	 ver_lr_scale : 1.0
	 every_lr_epoch : 40
	 easy_epochs : 60
	 lr_decay_epochs : [360, 420, 440]
	 normal_epochs : 440
	 warmup_epochs : 0
	 warmup : False
	 warmup_init_lr : 0
	 milestones : [40, 70]
	 use_gm_after_epoch : 20
	 warmup_multi_step_epochs : 120
	 pretrain_new_params_epochs : 0
	 pretrain_new_params : False
	 epochs_per_val : 20
	 steps_per_log : 5
	 trial_run : False
	 phase : normal
	 resume : False
	 resume_epoch : 
	 resume_from : pretrained
	 cft : False
	 cft_iters : 1
	 cft_rho : 0.0008
 blocksetting :
	 backbone : 3
	 pool : 3
	 reduction : 8
	 classifier : 8
	 multi_seg : 3
 vis :
	 use : False
	 heat_map :
		 use : False
		 save_dir : D:/weights_results/HJL-ReID/heat_map
 device : cuda
Keys not found in source state_dict: 
	 num_batches_tracked  x53
Keys not found in destination state_dict: 
	 fc.weight
	 fc.bias
=> Loaded ImageNet Model: D:/weights_results/pretrained_model/imagenet_model\resnet50-19c8e357.pth
Model(
  (model): MDRS(
    (backbone): MDRSBackbone(
      (backbone): Sequential(
        (0): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace)
        (3): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        (4): Sequential(
          (0): Bottleneck(
            (conv1): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace)
            (downsample): Sequential(
              (0): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
          (1): Bottleneck(
            (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace)
          )
          (2): Bottleneck(
            (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace)
          )
        )
        (5): Sequential(
          (0): Bottleneck(
            (conv1): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace)
            (downsample): Sequential(
              (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
              (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
          (1): Bottleneck(
            (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace)
          )
          (2): Bottleneck(
            (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace)
          )
          (3): Bottleneck(
            (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace)
          )
        )
        (6): Bottleneck(
          (conv1): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace)
          (downsample): Sequential(
            (0): Conv2d(512, 1024, kernel_size=(1, 1), stride=(2, 2), bias=False)
            (1): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
      )
      (p0): Sequential(
        (0): Sequential(
          (0): Bottleneck(
            (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace)
          )
          (1): Bottleneck(
            (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace)
          )
          (2): Bottleneck(
            (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace)
          )
          (3): Bottleneck(
            (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace)
          )
          (4): Bottleneck(
            (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace)
          )
        )
        (1): Sequential(
          (0): Bottleneck(
            (conv1): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace)
            (downsample): Sequential(
              (0): Conv2d(1024, 2048, kernel_size=(1, 1), stride=(2, 2), bias=False)
              (1): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
          (1): Bottleneck(
            (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace)
          )
          (2): Bottleneck(
            (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace)
          )
        )
      )
      (p1): Sequential(
        (0): Sequential(
          (0): Bottleneck(
            (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace)
          )
          (1): Bottleneck(
            (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace)
          )
          (2): Bottleneck(
            (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace)
          )
          (3): Bottleneck(
            (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace)
          )
          (4): Bottleneck(
            (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace)
          )
        )
        (1): Sequential(
          (0): Bottleneck(
            (conv1): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace)
            (downsample): Sequential(
              (0): Conv2d(1024, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
          (1): Bottleneck(
            (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace)
          )
          (2): Bottleneck(
            (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace)
          )
        )
      )
      (p2): Sequential(
        (0): Sequential(
          (0): Bottleneck(
            (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace)
          )
          (1): Bottleneck(
            (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace)
          )
          (2): Bottleneck(
            (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace)
          )
          (3): Bottleneck(
            (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace)
          )
          (4): Bottleneck(
            (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace)
          )
        )
        (1): Sequential(
          (0): Bottleneck(
            (conv1): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace)
            (downsample): Sequential(
              (0): Conv2d(1024, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
          (1): Bottleneck(
            (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace)
          )
          (2): Bottleneck(
            (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace)
          )
        )
      )
    )
    (pool): MDRSPool(
      (pool0): AdaptiveMaxPool2d(output_size=(1, 1))
      (pool1): AdaptiveMaxPool2d(output_size=(2, 1))
      (pool2): AdaptiveMaxPool2d(output_size=(3, 1))
    )
    (reduction): MDRSReduction(
      (reduction): ModuleList(
        (0): Sequential(
          (0): Conv2d(2048, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU()
        )
        (1): Sequential(
          (0): Conv2d(2048, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU()
        )
        (2): Sequential(
          (0): Conv2d(2048, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU()
        )
        (3): Sequential(
          (0): Conv2d(2048, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU()
        )
        (4): Sequential(
          (0): Conv2d(2048, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU()
        )
        (5): Sequential(
          (0): Conv2d(2048, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU()
        )
        (6): Sequential(
          (0): Conv2d(2048, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU()
        )
        (7): Sequential(
          (0): Conv2d(2048, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU()
        )
      )
    )
    (multi_seg): MDRSMultiSeg(
      (deconv): ConvTranspose2d(2048, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1), bias=False)
      (deconv3): ConvTranspose2d(2048, 1024, kernel_size=(3, 3), stride=(3, 3), padding=(1, 1), output_padding=(2, 2), bias=False)
      (deconv3_pool): AdaptiveAvgPool2d(output_size=(24, 8))
      (deconv1): ConvTranspose2d(2048, 1024, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1), bias=False)
      (deconv2): ConvTranspose2d(1024, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1), bias=False)
      (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace)
      (conv): Conv2d(256, 8, kernel_size=(1, 1), stride=(1, 1))
    )
  )
)
Model size: 113.637 M
=> Loaded [model] from D:/weights_results/HJL-ReID/MDRS_ADAM_random_erasing_margin_0.3_cuhk_jpg_best\ckpt.pth, epoch 440, score:
cuhk03_np_detected_jpg -> cuhk03_np_detected_jpg[mAP:  75.6%], [cmc1:  78.1%], [cmc5:  90.5%], [cmc10:  93.9%]
Test...
Loading feat dict mat from D:/weights_results/HJL-ReID/MDRS_ADAM_random_erasing_margin_0.3_cuhk_jpg_best/MDRS_feat_dict.mat...
D:/weights_results/HJL-ReID/MDRS_ADAM_random_erasing_margin_0.3_cuhk_jpg_best/MDRS_feat_dict.mat has been existed.
cuhk03_np_detected_jpg -> cuhk03_np_detected_jpg[mAP:  75.6%], [cmc1:  78.1%], [cmc5:  90.5%], [cmc10:  93.9%]
destination path dir D:/weights_results/HJL-ReID/MDRS_ADAM_random_erasing_margin_0.3_cuhk_jpg_best has been already exist.
=> Checkpoint Saved to D:/weights_results/HJL-ReID/MDRS_ADAM_random_erasing_margin_0.3_cuhk_jpg_best\ckpt.pth
```

# Performances
## Market1501
| Methods | mAP |	Rank-1 | Rank-5 |	Rank-10 | 
|---|---|---|---|---|
| MDRS |	87.6 | 95.8 |	98.4 | 99.1 |
| Pyramid | 88.2 | 95.7 | 98.4 | 99.0 |
| DSA-reID | 87.6 | 95.7	| – |	– |
| MGN | 86.9 | 95.7 | – | – |
| PCB+triplet | 83.0 | 93.4 | 97.8 | 98.4 |
| CASN(PCB) | 82.8 | 94.4 | – | – |
| PCB+RPP | 81.6 | 93.8 | 97.5 | 98.5 |
| VPM | 80.8 | 93.0 | 97.8 | 98.8 | 
| PCB | 77.4 | 92.3 | 97.2 | 98.2 | 
| GLAD | 73.9 | 89.9 | – | – |
| MultiScale | 73.1 | 88.9 | – | – |
| PartLoss | 69.3 | 88.2 | – | – |
| PDC | 63.4 | 84.4 | – | – |
| MultiLoss | 64.4 | 83.9 | – | – |
| PAR | 63.4 | 81.0 | 92.0 | 94.7 |
| HydraPlus | – | 76.9 | 91.3 | 94.5 |
| MultiRegion | 41.2 | 66.4 | 85.0 | 90.2 |
| SPReID |	83.4 | 93.7 | 97.6 | 98.4 |
| AOS | 70.4 | 86.5 | – | – |
| Triplet Loss | 69.1 | 84.9 | 94.2 | – |
| Transfer | 65.5 | 83.7 | – | – |
| PAN | 63.4 | 82.8 | – | – |
| SVDNet | 62.1 | 82.3 | 92.3 | 95.2 |

## DukeMTMC-reID
| Methods | mAP |	Rank-1 | Rank-5 |	Rank-10 | 
|---|---|---|---|---|
| MDRS | 79.4 | 89.4 | 95.1 | 96.8 |
| Pyramid | 79.0 | 89.0 | 94.7 | 96.3 |
| MGN | 78.4 | 88.7 | – | – |
| CASN(PCB) | 73.7 | 87.7 | – | – |
| DSA-reID | 74.3 | 86.2 | – | – |
| SPReID | 73.3 | 86.0 | 93.0 | 94.5 |
| PCB+triplet | 73.2 | 84.1 | 92.4 | 94.5 |
| VPM | 72.6 | 83.6 | 91.7 | 94.2 |
| PCB+RPP | 69.2 | 83.3 | – | – |
| PSE+ECN | 75.7 | 84.5 | – | – |
| DNN + CRF | 69.5 | 84.9 | – | – |
| GP-reid | 72.8 | 85.2 | – | – |
| AOS | 62.1 | 79.2 | – | – |

## CUHK03
| – | Labelled | - | - | - | Detected | - | - | - |
|---|---|---|---|---|---|---|---|---|
| Methods | mAP |	Rank-1 | Rank-5 |	Rank-10 | mAP |	Rank-1 | Rank-5 |	Rank-10 | 
| MDRS | 76.4 | 79.0 | 91.1 | 94.6 | 74.2 | 78.7 | 90.5 | 94.1 |
| Pyramid | 76.9 | 78.9 | 91.0 | 94.4 | 74.8 | 78.9 | 90.7 | 94.5 |
| DSA-reID | 75.2 | 78.9 | – | – | 73.1 | 78.2 | – | – |
| CASN(PCB) | 68.0 | 73.7 | – | – | 64.4 | 71.5 | – | – |
| MGN | 67.4 | 68.0 | – | – | 66.0 | 68.0 | – | – |
| PCB+RPP | – | – | – | – | 57.5 | 63.7 | – | – |
| MLFN | 49.2 | 54.7 | – | – | 47.8 | 52.8 | – | – |
| AOS | – | – | – | – | 43.3 | 47.1 | – | – |
| SVDNet | 37.8 | 40.9 | – | – | 37.3 | 41.5 | – | – |
| PAN | 35.0 | 36.9 | – | – | 34.0 | 36.3 | – | – |
| IDE | 21.0 | 22.2 | – | – | 19.7 | 21.3 | – | – |


# Other models
- We have provided [PGFAReIDTrainer.py](MDRSREID/Trainer/PGFAReIDTrainer.py) and 
[HOReIDTrainer.py](MDRSREID/Trainer/HOReIDTrainer.py) for `PGFA` and `HOReID`.

# To Do
- Add instructions of other model `MGN`, `PCB`, `PGFA` and `HOReID`.
- I will consider if `MGNReID.py` and `PCBReIDTRainer.py` are necessary.
