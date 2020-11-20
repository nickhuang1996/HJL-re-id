from collections import OrderedDict
from MDRSREID.Loss_Meter.ID_loss import IDLoss
from MDRSREID.Loss_Meter.ID_smooth_loss import IDSmoothLoss
from MDRSREID.Loss_Meter.triplet_loss import TripletLoss
from MDRSREID.Loss_Meter.triplet_loss import TripletHardLoss
from MDRSREID.Loss_Meter.permutation_loss import PermutationLoss
from MDRSREID.Loss_Meter.verification_loss import VerificationLoss
from MDRSREID.Loss_Meter.PGFA_loss import PGFALoss
from MDRSREID.Loss_Meter.Seg_loss import SegLoss
from MDRSREID.Loss_Meter.Multi_Seg_loss import MultiSegLoss
from MDRSREID.Loss_Meter.Multi_Seg_GP_loss import MultiSegGPLoss
from MDRSREID.Loss_Meter.invariance_loss import InvNet


def loss_function_creation(cfg, tb_writer):
    loss_functions = OrderedDict()

    if cfg.id_loss.use:
        loss_functions[cfg.id_loss.name] = IDLoss(cfg.id_loss, tb_writer)
    if cfg.id_smooth_loss.use:
        cfg.id_smooth_loss.device = cfg.device
        cfg.id_smooth_loss.num_classes = cfg.model.num_classes  # cfg.model.num_classes
        loss_functions[cfg.id_smooth_loss.name] = IDSmoothLoss(cfg.id_smooth_loss, tb_writer)
    if cfg.tri_loss.use:
        loss_functions[cfg.tri_loss.name] = TripletLoss(cfg.tri_loss, tb_writer)
    if cfg.tri_hard_loss.use:
        loss_functions[cfg.tri_hard_loss.name] = TripletHardLoss(cfg.tri_hard_loss, tb_writer)
    if cfg.permutation_loss.use:
        cfg.permutation_loss.device = cfg.device
        loss_functions[cfg.permutation_loss.name] = PermutationLoss(cfg.permutation_loss, tb_writer)
    if cfg.verification_loss.use:
        loss_functions[cfg.verification_loss.name] = VerificationLoss(cfg.verification_loss, tb_writer)
    if cfg.pgfa_loss.use:
        loss_functions[cfg.pgfa_loss.name] = PGFALoss(cfg.pgfa_loss, tb_writer)
    if cfg.src_seg_loss.use:
        loss_functions[cfg.src_seg_loss.name] = SegLoss(cfg.src_seg_loss, tb_writer)
    if cfg.src_multi_seg_loss.use:
        loss_functions[cfg.src_multi_seg_loss.name] = MultiSegLoss(cfg.src_multi_seg_loss, tb_writer)
    if cfg.src_multi_seg_gp_loss.use:
        loss_functions[cfg.src_multi_seg_gp_loss.name] = MultiSegGPLoss(cfg.src_multi_seg_gp_loss, tb_writer)
    if cfg.inv_loss.use:
        cfg.inv_loss.device = cfg.device
        cfg.inv_loss.num_classes = cfg.dataset.train.target.num_classes
        loss_functions[cfg.inv_loss.name] = InvNet(cfg.inv_loss, tb_writer).to(cfg.device)
    return loss_functions
