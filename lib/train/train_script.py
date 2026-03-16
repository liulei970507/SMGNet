import os
# loss function related
from lib.utils.box_ops import giou_loss
from torch.nn.functional import l1_loss
from torch.nn import BCEWithLogitsLoss
# train pipeline related
from lib.train.trainers import LTRTrainer
# distributed training related
from torch.nn.parallel import DistributedDataParallel as DDP
# some more advanced functions
from .base_functions import *
# network related
from lib.models.tbsi_track import build_tbsi_track
from lib.models.qat_cls_weight_track import build_qat_cls_weight_track
from lib.models.qat_cls_weight_guide_track import build_qat_cls_weight_guide_track
from lib.models.qat_cls_weight_add_track import build_qat_cls_weight_add_track
from lib.models.qat_cls_weight_guide_trans_encoder_one_track import build_qat_cls_weight_guide_trans_encoder_one_track
from lib.models.qat_guide_trans_encoder_track import build_qat_guide_trans_encoder_track
from lib.models.qat_cls_weight_guide_trans_track import build_qat_cls_weight_guide_trans_track
from lib.models.qat_cls_weight_guide_trans_encoder_track import build_qat_cls_weight_guide_trans_encoder_track
from lib.models.qat_cls4_weight_guide_trans_encoder_track import build_qat_cls4_weight_guide_trans_encoder_track
from lib.models.qat_cls8_weight_guide_trans_encoder_track import build_qat_cls8_weight_guide_trans_encoder_track
from lib.models.qat_cls32_weight_guide_trans_encoder_track import build_qat_cls32_weight_guide_trans_encoder_track
from lib.models.qat_cls64_weight_guide_trans_encoder_track import build_qat_cls64_weight_guide_trans_encoder_track
from lib.models.qat_mlp_weight_guide_trans_encoder_track import build_qat_mlp_weight_guide_trans_encoder_track
from lib.models.qat_cls_weight_guide_trans_before_encoder_track import build_qat_cls_weight_guide_trans_before_encoder_track
from lib.models.qat_cls_weight_guide_tbsi_track import build_qat_cls_weight_guide_tbsi_track
from lib.models.qat_mlp_weight_track import build_qat_mlp_weight_track
from lib.models.qat_mlp_weight_guide_track import build_qat_mlp_weight_guide_track
from lib.models.qat_mlp_weight_guide_trans_track import build_qat_mlp_weight_guide_trans_track
# forward propagation related
from lib.train.actors import TBSITrackActor
from lib.train.actors import QATCLSTrackActor
from lib.train.actors import QATMLPTrackActor
from lib.train.actors import QATCLSDBTrackActor
from lib.train.actors import QATCLSDB25TrackActor
from lib.train.actors import QATCLSDB10TrackActor
from lib.train.actors import QATCLSDB50TrackActor
from lib.models.qat_cls_weight_film_track import build_qat_cls_weight_film_track
from lib.models.qat_cls_weight_bat_track import build_qat_cls_weight_bat_track
# for import modules
import importlib

from ..utils.focal_loss import FocalLoss


def run(settings):
    settings.description = 'Training script for TBSI RGB-T Tracker'

    # update the default configs with config file
    if not os.path.exists(settings.cfg_file):
        raise ValueError("%s doesn't exist." % settings.cfg_file)
    config_module = importlib.import_module("lib.config.%s.config" % settings.script_name)
    cfg = config_module.cfg
    config_module.update_config_from_file(settings.cfg_file)
    if settings.local_rank in [-1, 0]:
        print("New configuration is shown below.")
        for key in cfg.keys():
            print("%s configuration:" % key, cfg[key])
            print('\n')

    # update settings based on cfg
    update_settings(settings, cfg)

    # Record the training log
    log_dir = os.path.join(settings.save_dir, 'logs')
    if settings.local_rank in [-1, 0]:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
    settings.log_file = os.path.join(log_dir, "%s-%s.log" % (settings.script_name, settings.config_name))

    # Build dataloaders
    loader_train, loader_val = build_dataloaders(cfg, settings)

    if "RepVGG" in cfg.MODEL.BACKBONE.TYPE or "swin" in cfg.MODEL.BACKBONE.TYPE or "LightTrack" in cfg.MODEL.BACKBONE.TYPE:
        cfg.ckpt_dir = settings.save_dir

    # Create network
    if settings.script_name == "tbsi_track":
        net = build_tbsi_track(cfg)
    elif settings.script_name == "qat_cls_weight_track":
        net = build_qat_cls_weight_track(cfg)
    elif settings.script_name == "qat_cls_weight_guide_track":
        net = build_qat_cls_weight_guide_track(cfg)
    elif settings.script_name == "qat_cls_weight_add_track":
        net = build_qat_cls_weight_add_track(cfg)
    elif settings.script_name == "qat_cls_weight_guide_trans_track":
        net = build_qat_cls_weight_guide_trans_track(cfg)
    elif settings.script_name == "qat_cls_weight_guide_trans_encoder_one_track":
        net = build_qat_cls_weight_guide_trans_encoder_one_track(cfg)
    elif settings.script_name == "qat_guide_trans_encoder_track":
        net = build_qat_guide_trans_encoder_track(cfg)
    elif settings.script_name in ["qat_cls_weight_guide_trans_encoder_track", "qat_cls_weight_guide_trans_encoder_dbactor_track", "qat_cls_weight_guide_trans_encoder_db25actor_track", "qat_cls_weight_guide_trans_encoder_db50actor_track", "qat_cls_weight_guide_trans_encoder_db10actor_track"]:
        net = build_qat_cls_weight_guide_trans_encoder_track(cfg)
    elif settings.script_name == "qat_cls4_weight_guide_trans_encoder_track":
        net = build_qat_cls4_weight_guide_trans_encoder_track(cfg)
    elif settings.script_name == "qat_cls8_weight_guide_trans_encoder_track":
        net = build_qat_cls8_weight_guide_trans_encoder_track(cfg)
    elif settings.script_name == "qat_cls32_weight_guide_trans_encoder_track":
        net = build_qat_cls32_weight_guide_trans_encoder_track(cfg)
    elif settings.script_name == "qat_cls64_weight_guide_trans_encoder_track":
        net = build_qat_cls64_weight_guide_trans_encoder_track(cfg)
    elif settings.script_name == "qat_mlp_weight_guide_trans_encoder_track":
        net = build_qat_mlp_weight_guide_trans_encoder_track(cfg)
    elif settings.script_name == "qat_cls_weight_guide_trans_before_encoder_track":
        net = build_qat_cls_weight_guide_trans_before_encoder_track(cfg)
    elif settings.script_name == "qat_cls_weight_guide_tbsi_track":
        net = build_qat_cls_weight_guide_tbsi_track(cfg)
    elif settings.script_name == "qat_mlp_weight_track":
        net = build_qat_mlp_weight_track(cfg)
    elif settings.script_name == "qat_mlp_weight_guide_track":
        net = build_qat_mlp_weight_guide_track(cfg)
    elif settings.script_name == "qat_mlp_weight_guide_trans_track":
        net = build_qat_mlp_weight_guide_trans_track(cfg)
    elif settings.script_name == "qat_cls_weight_film_track":
        net = build_qat_cls_weight_film_track(cfg)
    elif settings.script_name == "qat_cls_weight_bat_track":
        net = build_qat_cls_weight_bat_track(cfg)
    else:
        raise ValueError("illegal script name")

    # wrap networks to distributed one
    net.cuda()
    if settings.local_rank != -1:
        # net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)  # add syncBN converter
        net = DDP(net, device_ids=[settings.local_rank], find_unused_parameters=True)
        settings.device = torch.device("cuda:%d" % settings.local_rank)
    else:
        settings.device = torch.device("cuda:1")
    settings.deep_sup = getattr(cfg.TRAIN, "DEEP_SUPERVISION", False)
    settings.distill = getattr(cfg.TRAIN, "DISTILL", False)
    settings.distill_loss_type = getattr(cfg.TRAIN, "DISTILL_LOSS_TYPE", "KL")
    # Loss functions and Actors
    if settings.script_name == "tbsi_track":
        focal_loss = FocalLoss()
        objective = {'giou': giou_loss, 'l1': l1_loss, 'focal': focal_loss, 'cls': BCEWithLogitsLoss()}
        loss_weight = {'giou': cfg.TRAIN.GIOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT, 'focal': 1., 'cls': 1.0}
        actor = TBSITrackActor(net=net, objective=objective, loss_weight=loss_weight, settings=settings, cfg=cfg)
    elif settings.script_name in ["qat_cls_weight_track", "qat_guide_trans_encoder_track", "qat_cls_weight_guide_trans_encoder_one_track", "qat_cls_weight_add_track", "qat_cls_weight_film_track", "qat_cls_weight_bat_track","qat_cls_weight_guide_track", "qat_cls_weight_guide_trans_track", "qat_cls_weight_guide_trans_encoder_track", "qat_cls4_weight_guide_trans_encoder_track", "qat_cls8_weight_guide_trans_encoder_track", "qat_cls32_weight_guide_trans_encoder_track", "qat_cls64_weight_guide_trans_encoder_track", "qat_cls_weight_guide_trans_before_encoder_track", "qat_cls_weight_guide_tbsi_track"]:
        focal_loss = FocalLoss()
        objective = {'giou': giou_loss, 'l1': l1_loss, 'focal': focal_loss, 'cls': BCEWithLogitsLoss(), 'p_tag': BCEWithLogitsLoss()}
        loss_weight = {'giou': cfg.TRAIN.GIOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT, 'focal': 1., 'cls': 1.0, 'p_tag': cfg.TRAIN.P_WEIGHT}
        print('loss_weight:', loss_weight)
        actor = QATCLSTrackActor(net=net, objective=objective, loss_weight=loss_weight, settings=settings, cfg=cfg)
    elif settings.script_name in ["qat_cls_weight_guide_trans_encoder_dbactor_track"]:
        focal_loss = FocalLoss()
        objective = {'giou': giou_loss, 'l1': l1_loss, 'focal': focal_loss, 'cls': BCEWithLogitsLoss(), 'p_tag': BCEWithLogitsLoss()}
        loss_weight = {'giou': cfg.TRAIN.GIOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT, 'focal': 1., 'cls': 1.0, 'p_tag': cfg.TRAIN.P_WEIGHT}
        print('loss_weight:', loss_weight)
        actor = QATCLSDBTrackActor(net=net, objective=objective, loss_weight=loss_weight, settings=settings, cfg=cfg)
    elif settings.script_name in ["qat_cls_weight_guide_trans_encoder_db25actor_track"]:
        focal_loss = FocalLoss()
        objective = {'giou': giou_loss, 'l1': l1_loss, 'focal': focal_loss, 'cls': BCEWithLogitsLoss(), 'p_tag': BCEWithLogitsLoss()}
        loss_weight = {'giou': cfg.TRAIN.GIOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT, 'focal': 1., 'cls': 1.0, 'p_tag': cfg.TRAIN.P_WEIGHT}
        print('loss_weight:', loss_weight)
        actor = QATCLSDB25TrackActor(net=net, objective=objective, loss_weight=loss_weight, settings=settings, cfg=cfg)
    elif settings.script_name in ["qat_cls_weight_guide_trans_encoder_db50actor_track"]:
        focal_loss = FocalLoss()
        objective = {'giou': giou_loss, 'l1': l1_loss, 'focal': focal_loss, 'cls': BCEWithLogitsLoss(), 'p_tag': BCEWithLogitsLoss()}
        loss_weight = {'giou': cfg.TRAIN.GIOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT, 'focal': 1., 'cls': 1.0, 'p_tag': cfg.TRAIN.P_WEIGHT}
        print('loss_weight:', loss_weight)
        actor = QATCLSDB50TrackActor(net=net, objective=objective, loss_weight=loss_weight, settings=settings, cfg=cfg)
    elif settings.script_name in ["qat_cls_weight_guide_trans_encoder_db10actor_track"]:
        focal_loss = FocalLoss()
        objective = {'giou': giou_loss, 'l1': l1_loss, 'focal': focal_loss, 'cls': BCEWithLogitsLoss(), 'p_tag': BCEWithLogitsLoss()}
        loss_weight = {'giou': cfg.TRAIN.GIOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT, 'focal': 1., 'cls': 1.0, 'p_tag': cfg.TRAIN.P_WEIGHT}
        print('loss_weight:', loss_weight)
        actor = QATCLSDB10TrackActor(net=net, objective=objective, loss_weight=loss_weight, settings=settings, cfg=cfg)
    elif settings.script_name in ["qat_mlp_weight_track", "qat_mlp_weight_guide_trans_encoder_track", "qat_mlp_weight_guide_track", "qat_mlp_weight_guide_trans_track"]:
        focal_loss = FocalLoss()
        objective = {'giou': giou_loss, 'l1': l1_loss, 'focal': focal_loss, 'cls': BCEWithLogitsLoss(), 'p_tag': l1_loss}
        loss_weight = {'giou': cfg.TRAIN.GIOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT, 'focal': 1., 'cls': 1.0, 'p_tag': cfg.TRAIN.P_WEIGHT}
        print('loss_weight:', loss_weight)
        actor = QATMLPTrackActor(net=net, objective=objective, loss_weight=loss_weight, settings=settings, cfg=cfg)
    else:
        raise ValueError("illegal script name")

    # if cfg.TRAIN.DEEP_SUPERVISION:
    #     raise ValueError("Deep supervision is not supported now.")

    # Optimizer, parameters, and learning rates
    optimizer, lr_scheduler = get_optimizer_scheduler(net, cfg)
    use_amp = getattr(cfg.TRAIN, "AMP", False)
    trainer = LTRTrainer(actor, [loader_train, loader_val], optimizer, settings, lr_scheduler, use_amp=use_amp)

    # train process
    trainer.train(cfg.TRAIN.EPOCH, load_latest=True, fail_safe=True)
