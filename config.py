# --------------------------------------------------------
# Focal Modulation Network
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Jianwei Yang (jianwyan@microsoft.com)
# Based on Swin Transformer written by Zhe Liu
# --------------------------------------------------------

import os
import yaml
from yacs.config import CfgNode as CN

_C = CN()

# Base config files
_C.BASE = ['']

# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()
# Batch size for a single GPU, could be overwritten by command line argument
_C.DATA.ROOT = '/PATH/TO/videos'
_C.DATA.TRAIN_FILE = '/PATH/TO/train.txt'
_C.DATA.VAL_FILE = '/PATH/TO/val.txt'
_C.DATA.NUM_FRAMES = 32
_C.DATA.NUM_CLASSES = 101
_C.DATA.LABEL_LIST = 'labels/ucf_101_labels.csv'
_C.DATA.BATCH_SIZE = 128
_C.DATA.TEST_BATCH_SIZE = 128

# Path to dataset, could be overwritten by command line argument
_C.DATA.DATA_PATH = ''
# Dataset name
_C.DATA.DATASET = 'imagenet'
# Input image size
_C.DATA.IMG_SIZE = 224
_C.DATA.INPUT_SIZE = 224
# Interpolation to resize image (random, bilinear, bicubic)
_C.DATA.INTERPOLATION = 'bicubic'
# Use zipped dataset instead of folder dataset
# could be overwritten by command line argument
_C.DATA.ZIP_MODE = False
# Cache Data in Memory, could be overwritten by command line argument
_C.DATA.CACHE_MODE = 'part'
# Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.
_C.DATA.PIN_MEMORY = True
# Number of data loading threads
_C.DATA.NUM_WORKERS = 8

##-------------------------------------------------------
#UniformerV2 data settings
#---------------------------------------------------------

# The separator used between path and label.
_C.DATA.PATH_LABEL_SEPARATOR = " "

# Video path prefix if any.
_C.DATA.PATH_PREFIX = ""

# The list of video path prefix if any.
_C.DATA.PATH_PREFIX_LIST = [""]

# Label file path template.
_C.DATA.LABEL_PATH_TEMPLATE = "somesomev1_rgb_{}_split.txt"

# Label file path template.
_C.DATA.IMAGE_TEMPLATE = "{:05d}.jpg"

# The video sampling rate of the input clip.
_C.DATA.SAMPLING_RATE = 8

# Eigenvalues for PCA jittering. Note PCA is RGB based.
_C.DATA.TRAIN_PCA_EIGVAL = [0.225, 0.224, 0.229]

# Eigenvectors for PCA jittering.
_C.DATA.TRAIN_PCA_EIGVEC = [
    [-0.5675, 0.7192, 0.4009],
    [-0.5808, -0.0045, -0.8140],
    [-0.5836, -0.6948, 0.4203],
]

# If a imdb have been dumpped to a local file with the following format:
# `{"im_path": im_path, "class": cont_id}`
# then we can skip the construction of imdb and load it from the local file.
_C.DATA.PATH_TO_PRELOAD_IMDB = ""

# The mean value of the video raw pixels across the R G B channels.
_C.DATA.MEAN = [0.45, 0.45, 0.45]
# List of input frame channel dimensions.

_C.DATA.INPUT_CHANNEL_NUM = [3, 3]

# The std value of the video raw pixels across the R G B channels.
_C.DATA.STD = [0.225, 0.225, 0.225]

# The spatial augmentation jitter scales for training.
_C.DATA.TRAIN_JITTER_SCALES = [256, 320]

# The relative scale range of Inception-style area based random resizing augmentation.
# If this is provided, DATA.TRAIN_JITTER_SCALES above is ignored.
_C.DATA.TRAIN_JITTER_SCALES_RELATIVE = []

# The relative aspect ratio range of Inception-style area based random resizing
# augmentation.
_C.DATA.TRAIN_JITTER_ASPECT_RELATIVE = []

# If True, perform stride length uniform temporal sampling.
_C.DATA.USE_OFFSET_SAMPLING = False

# Whether to apply motion shift for augmentation.
_C.DATA.TRAIN_JITTER_MOTION_SHIFT = False

# The spatial crop size for training.
_C.DATA.TRAIN_CROP_SIZE = 224

# The spatial crop size for testing.
_C.DATA.TEST_CROP_SIZE = 256

# Input videos may has different fps, convert it to the target video fps before
# frame sampling.
_C.DATA.TARGET_FPS = 30

# Decoding backend, options include `pyav` or `torchvision`
_C.DATA.DECODING_BACKEND = "pyav"

# if True, sample uniformly in [1 / max_scale, 1 / min_scale] and take a
# reciprocal to get the scale. If False, take a uniform sample from
# [min_scale, max_scale].
_C.DATA.INV_UNIFORM_SAMPLE = False

# If True, perform random horizontal flip on the video frames during training.
_C.DATA.RANDOM_FLIP = True

# If True, calculdate the map as metric.
_C.DATA.MULTI_LABEL = False

# Method to perform the ensemble, options include "sum" and "max".
_C.DATA.ENSEMBLE_METHOD = "sum"

# If True, revert the default input channel (RBG <-> BGR).
_C.DATA.REVERSE_INPUT_CHANNEL = False

# Whether read from mc.
_C.DATA.MC = False

# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Model type
_C.MODEL.TYPE = 'focal'
# Model name
_C.MODEL.NAME = ''
# Checkpoint to resume, could be overwritten by command line argument
_C.MODEL.RESUME = ''
# Number of classes, overwritten in data preparation
_C.MODEL.NUM_CLASSES = 1000
# Dropout rate
_C.MODEL.DROP_RATE = 0.0
# Drop path rate
_C.MODEL.DROP_PATH_RATE = 0.1
# Label Smoothing
_C.MODEL.LABEL_SMOOTHING = 0.1
# Whether load pretrained model
_C.MODEL.PRETRAINED = False
# Mode specific
_C.MODEL.SPEC = CN(new_allowed=True)

# FocalNet parameters
_C.MODEL.FOCAL = CN()
_C.MODEL.FOCAL.PATCH_SIZE = 4
_C.MODEL.FOCAL.IN_CHANS = 3
_C.MODEL.FOCAL.EMBED_DIM = 96
_C.MODEL.FOCAL.DEPTHS = [2, 2, 6, 2]
_C.MODEL.FOCAL.MLP_RATIO = 4.
_C.MODEL.FOCAL.PATCH_NORM = True
_C.MODEL.FOCAL.FOCAL_LEVELS = [2, 2, 2, 2]
_C.MODEL.FOCAL.FOCAL_WINDOWS = [3, 3, 3, 3]
_C.MODEL.FOCAL.FOCAL_FACTORS = [2, 2, 2, 2]
_C.MODEL.FOCAL.USE_CONV_EMBED = False
_C.MODEL.FOCAL.USE_LAYERSCALE = False
_C.MODEL.FOCAL.USE_POSTLN = False
_C.MODEL.FOCAL.USE_POSTLN_IN_MODULATION = False
_C.MODEL.FOCAL.NORMALIZE_MODULATOR = False

# tube embedding
_C.MODEL.TUBELET_SIZE = 1
# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.START_EPOCH = 0
_C.TRAIN.EPOCHS = 300
_C.TRAIN.WARMUP_EPOCHS = 20
_C.TRAIN.WEIGHT_DECAY = 0.05
_C.TRAIN.BASE_LR = 5e-4
_C.TRAIN.WARMUP_LR = 5e-7
_C.TRAIN.MIN_LR = 5e-6
# Clip gradient norm
_C.TRAIN.CLIP_GRAD = 5.0
# Auto resume from latest checkpoint
_C.TRAIN.AUTO_RESUME = True
# Gradient accumulation steps
# could be overwritten by command line argument
_C.TRAIN.ACCUMULATION_STEPS = 0
# Whether to use gradient checkpointing to save memory
# could be overwritten by command line argument
_C.TRAIN.USE_CHECKPOINT = False

# LR scheduler
_C.TRAIN.LR_SCHEDULER = CN()
_C.TRAIN.LR_SCHEDULER.NAME = 'cosine'
# Epoch interval to decay LR, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_EPOCHS = 30
# LR decay rate, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_RATE = 0.1

# Optimizer
_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.NAME = 'adamw'
# Optimizer Epsilon
_C.TRAIN.OPTIMIZER.EPS = 1e-8
# Optimizer Betas
_C.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999)
# SGD momentum
_C.TRAIN.OPTIMIZER.MOMENTUM = 0.9
# pretrained path
_C.TRAIN.PRETRAINED_PATH = ''

# -----------------------------------------------------------------------------
# Augmentation settings
# -----------------------------------------------------------------------------
_C.AUG = CN()
# Color jitter factor
_C.AUG.COLOR_JITTER = 0.4
# Use AutoAugment policy. "v0" or "original"
_C.AUG.AUTO_AUGMENT = 'rand-m9-mstd0.5-inc1'
# Random erase prob
_C.AUG.REPROB = 0.25
# Random erase mode
_C.AUG.REMODE = 'pixel'
# Random erase count
_C.AUG.RECOUNT = 1
# Mixup alpha, mixup enabled if > 0
_C.AUG.MIXUP = 0.8
# Cutmix alpha, cutmix enabled if > 0
_C.AUG.CUTMIX = 1.0
# Cutmix min/max ratio, overrides alpha and enables cutmix if set
_C.AUG.CUTMIX_MINMAX = None
# Probability of performing mixup or cutmix when either/both is enabled
_C.AUG.MIXUP_PROB = 1.0
# Probability of switching to cutmix when both mixup and cutmix enabled
_C.AUG.MIXUP_SWITCH_PROB = 0.5
# How to apply mixup/cutmix params. Per "batch", "pair", or "elem"
_C.AUG.MIXUP_MODE = 'batch'
_C.AUG.GRAY_SCALE = 0.2

# flip ratio
_C.AUG.FLIP_RATIO = 0.5
# frame interval
_C.AUG.FRAME_INTERVAL = 2
_C.AUG.LABEL_SMOOTH = 0.1
# _C.AUG.COLOR_JITTER = 0.8
# _C.AUG.MIXUP = 0.8
# _C.AUG.CUTMIX = 1.0
# _C.AUG.MIXUP_SWITCH_PROB = 0.5

# -----------------------------------------------------------------------------
# Testing settings
# -----------------------------------------------------------------------------
_C.TEST = CN()
# Whether to use center crop when testing
_C.TEST.CROP = True
_C.TEST.NUM_CLIP = 1
_C.TEST.NUM_CROP = 1
_C.TEST.ONLY_TEST = False

# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------
# Mixed precision opt level, if O0, no amp is used ('O0', 'O1', 'O2')
# overwritten by command line argument
_C.AMP_OPT_LEVEL = ''
# Path to output folder, overwritten by command line argument
_C.OUTPUT = ''
# Tag of experiment, overwritten by command line argument
_C.TAG = 'default'
# Frequency to save checkpoint
_C.SAVE_FREQ = 1
# Frequency to logging info
_C.PRINT_FREQ = 100
# Fixed random seed
_C.SEED = 0
# Perform evaluation only, overwritten by command line argument
_C.EVAL_MODE = False
# Test throughput only, overwritten by command line argument
_C.THROUGHPUT_MODE = False
# Debug only so that skip dataloader initialization, overwritten by command line argument
_C.DEBUG_MODE = False
# local rank for DistributedDataParallel, given by command line argument
_C.LOCAL_RANK = 0

# -----------------------------------------------------------------------------
# UNIFORMERV2 options
# -----------------------------------------------------------------------------

_C.UNIFORMERV2 = CN()
# backbone for UniFormerV2, vit_b32, vit_b16, bit_l14
_C.UNIFORMERV2.BACKBONE = 'vit_b16'

# feature layers
_C.UNIFORMERV2.N_LAYERS = 4

# feature dimension
_C.UNIFORMERV2.N_DIM = 768

# head number
_C.UNIFORMERV2.N_HEAD = 12

# MLP ratio
_C.UNIFORMERV2.MLP_FACTOR = 4.0

# backbone droppath rate
_C.UNIFORMERV2.BACKBONE_DROP_PATH_RATE = 0.0

# droppath rate
_C.UNIFORMERV2.DROP_PATH_RATE = 0.0

# MLP dropout
_C.UNIFORMERV2.MLP_DROPOUT = [0.5, 0.5, 0.5, 0.5]

# CLS layer dropout
_C.UNIFORMERV2.CLS_DROPOUT = 0.5

# index list of return features
_C.UNIFORMERV2.RETURN_LIST = [8, 9, 10, 11]

# local block reduction
_C.UNIFORMERV2.DW_REDUCTION = 1.5

# whether add temporal downsample
_C.UNIFORMERV2.TEMPORAL_DOWNSAMPLE = True

# whether use local MHRA
_C.UNIFORMERV2.NO_LMHRA = False

# local block number
_C.UNIFORMERV2.DOUBLE_LMHRA = True

# pretrained model for UniFormerV2
_C.UNIFORMERV2.PRETRAINED_PATH = ''

# delete pretrained head
_C.UNIFORMERV2.DELETE_SPECIAL_HEAD = False

# freeze backbone
_C.UNIFORMERV2.FROZEN = False

# ---------------------------------------------------------------------------- #
# Optimizer options
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()

# Base learning rate.
_C.SOLVER.BASE_LR = 0.1

# Learning rate policy (see utils/lr_policy.py for options and examples).
_C.SOLVER.LR_POLICY = "cosine"

# Final learning rates for 'cosine' policy.
_C.SOLVER.COSINE_END_LR = 0.0

# Exponential decay factor.
_C.SOLVER.GAMMA = 0.1

# Step size for 'exp' and 'cos' policies (in epochs).
_C.SOLVER.STEP_SIZE = 1

# Steps for 'steps_' policies (in epochs).
_C.SOLVER.STEPS = []

# Learning rates for 'steps_' policies.
_C.SOLVER.LRS = []

# Maximal number of epochs.
_C.SOLVER.MAX_EPOCH = 300

# Momentum.
_C.SOLVER.MOMENTUM = 0.9

# Momentum dampening.
_C.SOLVER.DAMPENING = 0.0

# Nesterov momentum.
_C.SOLVER.NESTEROV = True

# L2 regularization.
_C.SOLVER.WEIGHT_DECAY = 1e-4

# Start the warm up from SOLVER.BASE_LR * SOLVER.WARMUP_FACTOR.
_C.SOLVER.WARMUP_FACTOR = 0.1

# Gradually warm up the SOLVER.BASE_LR over this number of epochs.
_C.SOLVER.WARMUP_EPOCHS = 0.0

# The start learning rate of the warm up.
_C.SOLVER.WARMUP_START_LR = 0.01

# Optimization method.
_C.SOLVER.OPTIMIZING_METHOD = "sgd"

# Base learning rate is linearly scaled with NUM_SHARDS.
_C.SOLVER.BASE_LR_SCALE_NUM_SHARDS = False

# If True, start from the peak cosine learning rate after warm up.
_C.SOLVER.COSINE_AFTER_WARMUP = False

# If True, perform no weight decay on parameter with one dimension (bias term, etc).
_C.SOLVER.ZERO_WD_1D_PARAM = False

# gradient norm clipping.
_C.SOLVER.CLIP_GRADIENT = 20

# backbone lr ratio.
_C.SOLVER.BACKBONE_LR_RATIO = 0.1

# special parameter list.
_C.SOLVER.SPECIAL_LIST = []

# special parameter list.
_C.SOLVER.SPECIAL_RATIO = 1.


def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg)
            )
    print('=> merge config from {}'.format(cfg_file))
    config.merge_from_file(cfg_file)
    config.freeze()


def update_config(config, args):
    _update_config_from_file(config, args.cfg)

    config.defrost()
    if args.opts:
        config.merge_from_list(args.opts)

    # merge from specific arguments
    if args.batch_size:
        config.DATA.BATCH_SIZE = args.batch_size
    if args.resume:
        config.MODEL.RESUME = args.resume
    if args.accumulation_steps:
        config.TRAIN.ACCUMULATION_STEPS = args.accumulation_steps
    if args.use_checkpoint:
        config.TRAIN.USE_CHECKPOINT = True
    if args.amp_opt_level:
        config.AMP_OPT_LEVEL = args.amp_opt_level
    if args.output:
        config.OUTPUT = args.output
    if args.eval:
        config.EVAL_MODE = True

    # set local rank for distributed training
    config.LOCAL_RANK = args.local_rank

    # output folder
    config.TAG = "LR_{}_EP_{}_OPT_{}_FRAMES_{}".format(config.TRAIN.BASE_LR, config.TRAIN.EPOCHS,
                                                       config.TRAIN.OPTIMIZER.NAME, config.DATA.NUM_FRAMES)
    config.OUTPUT = os.path.join(config.OUTPUT, config.DATA.DATASET, config.MODEL.NAME, config.TAG)

    config.freeze()


def get_config(args):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    update_config(config, args)

    return config
