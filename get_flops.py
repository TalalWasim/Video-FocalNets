import os
import time
import argparse
import datetime
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy
from utils import AverageMeter
from datasets.blending import CutmixMixupBlending
from config import get_config
from classification import build_model
from datasets.build import build_dataloader
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from logger import create_logger
from utils import load_checkpoint, save_checkpoint, get_grad_norm, auto_resume_helper, reduce_tensor
from timm.models.layers import trunc_normal_

from thop import profile, clever_format

device = "cuda:0" if torch.cuda.is_available() else "cpu"



def parse_option():
    parser = argparse.ArgumentParser('FocalNet training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, required=False, metavar="FILE", help='path to config file',
                        default='./configs/kinetics400/video-focalnet_tiny.yaml')
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument("--local_rank", type=int, default=0, help='local rank for DistributedDataParallel')

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config



_, config = parse_option()
config.defrost()
config.DATA.NUM_FRAMES = 8
config.freeze()
model = build_model(config)
model = model.to(device)
data = torch.randn(1,8,3,224,224).to(device)

macs, params = profile(model, inputs=(data, ))
macs, _ = clever_format([macs, params], "%.3f")

print("gflops:", macs)