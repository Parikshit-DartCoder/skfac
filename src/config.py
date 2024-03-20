"""
network config setting, will be used in train.py and eval.py
"""
from easydict import EasyDict as ed

# config for resnet50, imagenet2012, GPU
config_gpu = ed({
    "class_num": 1001,
    "batch_size": 32,
    "loss_scale": 1,
    "momentum": 0.9,
    "weight_decay": 5e-4,
    "epoch_size": 50,
    "save_checkpoint": True,
    "save_checkpoint_epochs": 1,
    "keep_checkpoint_max": 50,
    "save_checkpoint_path": "./checkpoint_skfac",
    "use_label_smooth": True,
    "label_smooth_factor": 0.1,
    "lr_init": 0.00625,
    "lr_decay": 0.87,
    "lr_end_epoch": 50,
    "damping_init": 0.03,
    "frequency": 834,
})
