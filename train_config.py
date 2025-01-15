

import os
import numpy as np
from easydict import EasyDict as edict

config = edict()
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config.TRAIN = edict()
#### below are params for dataiter
config.TRAIN.process_num = 4

############

config.TRAIN.batch_size = 32
config.TRAIN.validatiojn_batch_size = config.TRAIN.batch_size
config.TRAIN.accumulation_batch_size = 32
config.TRAIN.log_interval = 10                  ##10 iters for a log msg
config.TRAIN.test_interval = 1
config.TRAIN.epoch = 15

config.TRAIN.init_lr=0.001
config.TRAIN.lr_scheduler='cos'        ### cos or ReduceLROnPlateau


if config.TRAIN.lr_scheduler=='ReduceLROnPlateau':
    config.TRAIN.epoch = 100
    config.TRAIN.lr_scheduler_factor = 0.1

config.TRAIN.weight_decay_factor = 2.e-2                                  ####l2
config.TRAIN.vis=False
#### if to check the training data

config.TRAIN.warmup_step=800
config.TRAIN.opt='Adamw'
config.TRAIN.SWA=-1    ### -1 use no swa   from which epoch start SWA
config.TRAIN.gradient_clip=-5




config.TRAIN.vis_mixcut=False
if config.TRAIN.vis:
    config.TRAIN.mix_precision=False                                            ##use mix precision to speedup, tf1.14 at least
else:
    config.TRAIN.mix_precision = True

config.TRAIN.opt='Adamw'
config.TRAIN.stage=1
config.MODEL = edict()



config.MODEL.model_path = './models/'                                        ## save directory

config.DATA = edict()

config.DATA.data_file='./train.csv'

config.DATA.data_root_path='../hms-harmful-brain-activity-classification/'





config.MODEL.early_stop=10

config.MODEL.pretrained_model=None


config.SEED=1086


from lib.utils.seed_utils import seed_everything

seed_everything(config.SEED)