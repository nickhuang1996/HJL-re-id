import torch
import time
from torch.nn.parallel import DataParallel
import os.path as osp
from copy import deepcopy

from MDRSREID.Trainer.pre_initialization import pre_initialization
from MDRSREID.Trainer.dataloader_creation import dataloader_creation
from MDRSREID.Trainer.model_creation import model_creation
from MDRSREID.Trainer.optimizer_creation import optimizer_creation
from MDRSREID.Trainer.lr_scheduler_creation import lr_scheduler_creation
from MDRSREID.Trainer.evaluation_creation import evaluation_creation
from MDRSREID.Trainer.loss_function_creation import loss_function_creation
from MDRSREID.Trainer.analyze_function_creation import analyze_function_creation
from MDRSREID.Trainer.print_log import print_log as print_step_log

from MDRSREID.utils.load_state_dict import load_state_dict
from MDRSREID.utils.device_utils.recursive_to_device import recursive_to_device
from MDRSREID.utils.log_utils.log import join_str
from MDRSREID.utils.log_utils.log import score_str
from MDRSREID.utils.log_utils.log import write_to_file
from MDRSREID.utils.may_make_dirs import may_make_dirs


class HOReIDTrainer(object):
    def __init__(self, cfg):
        self.cfg = cfg

        self.current_ep = 0
        # Init the test loader
        self.test_loader = dataloader_creation(self.cfg, mode='test', domain='source', train_type='Supervised')
        if self.cfg.only_test is False:
            # TensorBoard object must not be in EasyDict()!!!!
            # cfg.log.tb_writer should be error!!!!
            if self.cfg.log.use_tensorboard:
                from tensorboardX import SummaryWriter
                self.tb_writer = SummaryWriter(log_dir=osp.join(self.cfg.log.exp_dir, 'tensorboard'))
            else:
                self.tb_writer = None

            self.source_train_loader = dataloader_creation(self.cfg, mode='train', domain='source',
                                                           train_type='Supervised')

            self.model = model_creation(self.cfg)
            self.optimizer = optimizer_creation(cfg, self.model)
            self.lr_scheduler = lr_scheduler_creation(cfg, self.optimizer, self.source_train_loader)
            self.loss_functions = loss_function_creation(cfg, self.tb_writer)
            self.analyze_functions = analyze_function_creation(cfg, self.tb_writer)

            self.epoch_start_time = 0
            self.trial_run_steps = 3 if cfg.optim.trial_run else None

            self.current_step = 0  # will NOT be reset between epochs
            self.steps_per_log = self.cfg.optim.steps_per_log
            self.print_step_log = print_step_log

            self.current_ep = 0
            self.print_ep_log = None  # function
            self.eps_per_log = 1

            self.save_ckpt = {
                'model': self.model,
                'optimizer': self.optimizer,
                'lr_scheduler': self.lr_scheduler
            }

        else:
            # Init the test part
            self.model = model_creation(self.cfg)
            self.current_ep, _ = self.may_load_ckpt()
        if self.cfg.optim.resume is True:
            self.resume()


    def train(self):
        # dataset_sizes = len(self.source_train_loader.dataset)
        # class_num = self.source_train_loader.dataset.num_ids
        # assert self.cfg.model.num_classes == class_num, "cfg.model.num_classes should be {} in create_train_dataloader.py".format(class_num)

        print("End epoch is:", self.cfg.optim.epochs)
        for epoch in range(self.current_ep, self.cfg.optim.epochs):
            self.epoch_start_time = time.time()
            self.model.set_train_mode(fix_ft_layers=self.cfg.optim.phase == 'pretrain')

            for index, item in enumerate(self.source_train_loader):
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()
                self.optimizer.zero_grad()

                # Source item
                item = recursive_to_device(item, self.cfg.device)
                pred = self.model.forward(item, cfg=cfg, forward_type='Supervised')

                loss = 0
                if self.cfg.id_smooth_loss.use is True:
                    loss += self.loss_functions[self.cfg.id_smooth_loss.name](item, pred, step=self.current_step, is_gcned=False)['loss']
                    loss += self.loss_functions[self.cfg.id_smooth_loss.name](item, pred, step=self.current_step, is_gcned=True)['loss']
                if self.cfg.tri_hard_loss.use is True:
                    loss += self.loss_functions[self.cfg.tri_hard_loss.name](item, pred, step=self.current_step, is_gcned=False)['loss']
                    loss += self.loss_functions[self.cfg.tri_hard_loss.name](item, pred, step=self.current_step, is_gcned=True)['loss']
                if self.current_step >= len(self.source_train_loader) * self.cfg.optim.use_gm_after_epoch:
                    if self.cfg.permutation_loss.use is True:
                        loss += self.loss_functions[self.cfg.permutation_loss.name](item, pred, step=self.current_step, pos_neg=True)['loss']
                        # loss += self.loss_functions[self.cfg.permutation_loss.name](item, pred, step=self.current_step, pos_neg=False)['loss']
                    if self.cfg.verification_loss.use:
                        loss += self.loss_functions[self.cfg.verification_loss.name](item, pred, step=self.current_step)['loss']
                if self.cfg.analyze_computer.use is True:
                    self.analyze_functions[self.cfg.analyze_computer.name](item, pred, step=self.current_step, is_gcned=False)
                    self.analyze_functions[self.cfg.analyze_computer.name](item, pred, step=self.current_step, is_gcned=True)
                if self.cfg.verification_probability_analyze.use is True:
                    self.analyze_functions[self.cfg.verification_probability_analyze.name](item, pred, step=self.current_step, pos_neg=True)
                    self.analyze_functions[self.cfg.verification_probability_analyze.name](item, pred, step=self.current_step, pos_neg=False)
                if isinstance(loss, torch.Tensor):
                    loss.backward()
                self.optimizer.step()

                if ((self.current_step + 1) % self.steps_per_log == 0) and (self.print_step_log is not None):
                    self.print_step_log(self.cfg,
                                        self.current_ep,
                                        self.current_step,
                                        self.optimizer,
                                        self.loss_functions,
                                        self.analyze_functions,
                                        self.epoch_start_time)

                self.current_step += 1
                if (self.trial_run_steps is not None) and (index + 1 >= self.trial_run_steps):
                    break
            if ((self.current_ep + 1) % self.eps_per_log == 0) and (self.print_ep_log is not None):
                self.print_ep_log()
            self.current_ep += 1
            score_str = self.may_test()
            self.may_save_ckpt(score_str, self.current_ep)

    def _evaluation(self, score_strs, score_summary, test_dataset_name, test_dict, use_gcn, use_gm):
        score_dict = evaluation_creation(self.model_for_eval,
                                         test_dict['query'],
                                         test_dict['gallery'],
                                         deepcopy(self.cfg),
                                         use_gcn=use_gcn,
                                         use_gm=use_gm)
        score_strs.append(score_dict['scores_str'])
        score_summary.append("{}->{}: {} ({})".format(self.cfg.dataset.train.source.name,
                                                      test_dataset_name,
                                                      score_str(score_dict['cmc_scores'][0]).replace('%', ''),
                                                      score_str(score_dict['mAP']).replace('%', '')))

    def test(self):
        self.cfg.model_flow = 'test'
        print('======test======')

        score_strs = []
        score_summary = []
        for test_dataset_name, test_dict in self.test_loader.items():
            self.cfg.eval.test_feat_cache_file = osp.join(self.cfg.log.exp_dir, '{}_to_{}_feat_cache.pkl'.format(
                self.cfg.dataset.train.source.name, test_dataset_name))
            self.cfg.eval.score_prefix = '{} -> {}'.format(self.cfg.dataset.train.source.name, test_dataset_name).ljust(30)
            if self.cfg.only_test is True:
                self._evaluation(score_strs, score_summary, test_dataset_name, test_dict, use_gcn=False, use_gm=False)
                self._evaluation(score_strs, score_summary, test_dataset_name, test_dict, use_gcn=True, use_gm=False)
            self._evaluation(score_strs, score_summary, test_dataset_name, test_dict, use_gcn=True, use_gm=True)

        score_str_ = join_str(score_strs, '\n')
        score_summary = ('Epoch {}'.format(self.current_ep)).ljust(12) + ', '.join(score_summary) + '\n'
        write_to_file(self.cfg.log.score_file, score_summary, append=True)

        self.cfg.model_flow = 'train'
        return score_str_

    def may_test(self):
        score_str = ''
        # You can force not testing by manually setting dont_test=True.
        if not hasattr(self.cfg.optim, 'dont_test') or not self.cfg.optim.dont_test:
            if (self.current_ep % self.cfg.optim.epochs_per_val == 0) or (
                    self.current_ep == self.cfg.optim.epochs) or self.cfg.optim.trial_run:
                score_str = self.test()
        return score_str

    def may_save_ckpt(self, score, epoch):
        """
        :param score: mAP and CMC scores
        :param epoch:
        :return:
        """
        state_dicts = {}
        if not self.cfg.optim.trial_run:
            state_dicts = {
                key: item.state_dict()
                for key, item in self.save_ckpt.items()
                if item is not None
            }
        ckpt = dict(state_dicts=state_dicts,
                    epoch=epoch,
                    score=score)
        may_make_dirs(dst_path=osp.dirname(self.cfg.log.ckpt_file))
        torch.save(ckpt, self.cfg.log.ckpt_file)
        msg = '=> Checkpoint Saved to {}'.format(self.cfg.log.ckpt_file)
        print(msg)

    def may_load_ckpt(self, load_model=False, load_optimizer=False, load_lr_scheduler=False, strict=True):
        exp_dir = self.cfg.log.exp_dir  # D:/weights_results/HOReID/pre-trained
        resume_epoch = self.cfg.optim.resume_epoch
        # resume from the resume_test_epoch
        if cfg.optim.resume_from is 'pretrained':
            self.model.model.encoder.load_state_dict(
                torch.load(osp.join(exp_dir, 'encoder_{}.pkl'.format(resume_epoch))))
            self.model.model.bnclassifiers.load_state_dict(
                torch.load(osp.join(exp_dir, 'bnclassifiers_{}.pkl'.format(resume_epoch))))
            self.model.model.bnclassifiers2.load_state_dict(
                torch.load(osp.join(exp_dir, 'bnclassifiers2_{}.pkl'.format(resume_epoch))))
            self.model.model.graph_conv_net.gcn.load_state_dict(
                torch.load(osp.join(exp_dir, 'gcn_{}.pkl'.format(resume_epoch))))
            self.model.model.graph_matching_net.load_state_dict(
                torch.load(osp.join(exp_dir, 'gmnet_{}.pkl'.format(resume_epoch))))
            self.model.model.verificator.load_state_dict(
                torch.load(osp.join(exp_dir, 'verificator_{}.pkl'.format(resume_epoch))))
            return resume_epoch, None
        elif cfg.optim.resume_from is 'whole':
            ckpt_file = self.cfg.log.ckpt_file
            assert osp.exists(ckpt_file), "ckpt_file {} does not exist!".format(ckpt_file)
            assert osp.isfile(ckpt_file), "ckpt_file {} is not file!".format(ckpt_file)
            ckpt = torch.load(ckpt_file, map_location=(lambda storage, loc: storage))

            load_ckpt = {}
            if load_model:
                load_ckpt['model'] = self.model
            if load_optimizer:
                load_ckpt['optimizer'] = self.optimizer
            if load_lr_scheduler:
                load_ckpt['lr_scheduler'] = self.lr_scheduler

            for name, item in load_ckpt.items():
                if item is not None:
                    # Only nn.Module.load_state_dict has this keyword argument
                    if not isinstance(item, torch.nn.Module) or strict:
                        item.load_state_dict(ckpt['state_dicts'][name])
                    else:
                        load_state_dict(item, ckpt['state_dicts'][name])

            load_ckpt_str = ', '.join(load_ckpt.keys())
            msg = '=> Loaded [{}] from {}, epoch {}, score:\n{}'.format(load_ckpt_str, ckpt_file, ckpt['epoch'],
                                                                        ckpt['score'])
            print(msg)
            return ckpt['epoch'], ckpt['score']

    def resume(self):
        resume_ep, score = self.may_load_ckpt(load_model=True, load_optimizer=True)
        self.current_ep = resume_ep
        self.current_step = resume_ep * len(self.source_train_loader)

    @property
    def model_for_eval(self):
        # Due to an abnormal bug, I decide not to use DataParallel during testing.
        # The bug case: total im 15913, batch size 32, 15913 % 32 = 9, it's ok to use 2 gpus,
        # but when I used 4 gpus, it threw error at the last batch: [line 83, in parallel_apply
        # , ... TypeError: forward() takes at least 2 arguments (2 given)]
        return self.model.module if isinstance(self.model, DataParallel) else self.model


if __name__ == '__main__':
    cfg = pre_initialization()
    trainer = HOReIDTrainer(cfg)
    if cfg.only_test:
        trainer.test()
    else:
        trainer.train()
