import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src import config
from src.resnet_skfac import resnet50
from src.dataset import create_dataset
from src.crossentropy import CrossEntropy
from src.skfac import SKFAC_GPU as SKFAC

parser = argparse.ArgumentParser(description='Image classification')
parser.add_argument('--dataset_path', type=str, default='ImageNet2012/train', help='Dataset path')
args_opt = parser.parse_args()

# Set seed for reproducibility
torch.manual_seed(1)
torch.cuda.manual_seed(1)


def adjust_learning_rate(epoch, batch_idx):
    lr_adj = config.lr_decay ** (epoch + float(batch_idx + 1) / 40036)
    return config.lr_init * lr_adj


def adjust_damping(epoch, batch_idx):
    epoch += float(batch_idx + 1) / 40036
    return config.damping_init * (config.lr_decay ** (epoch / 10))


class Model_SKFAC(nn.Module):
    def __init__(self, net, loss_fn, optimizer, loss_scale_manager, metrics):
        super(Model_SKFAC, self).__init__()
        self.net = net
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.loss_scale_manager = loss_scale_manager
        self.metrics = metrics

    def forward(self, x):
        return self.net(x)


if __name__ == '__main__':
    target = 'GPU'
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'

    # Create dataset
    train_dataset = create_dataset(dataset_path=args_opt.dataset_path, do_train=True, repeat_num=1,
                                   batch_size=config.batch_size, target=target)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

    # Define net
    damping = get_model_damping_skfac(70)
    lr = get_model_lr_skfac(config.lr_end_epoch)
    net = resnet50(class_num=config.class_num, damping=damping, loss_scale=config.loss_scale,
                   frequency=config.frequency, batch_size=config.batch_size)

    # Define loss function
    loss_fn = CrossEntropy(smooth_factor=config.label_smooth_factor, num_classes=config.class_num)

    # Define optimizer
    optimizer = SKFAC(filter(lambda x: x.requires_grad, net.parameters()), Tensor(lr), config.momentum,
                      filter(lambda x: 'matrix_A' in x.name, net.parameters()),
                      filter(lambda x: 'matrix_G' in x.name, net.parameters()),
                      filter(lambda x: 'A_inv_max' in x.name, net.parameters()),
                      filter(lambda x: 'G_inv_max' in x.name, net.parameters()),
                      config.weight_decay, config.loss_scale)

    # Define loss scale manager
    loss_scale_manager = FixedLossScaleManager(config.loss_scale, drop_overflow_update=False)

    # Define metrics
    metrics = {'acc'}

    # Create model instance
    model = Model_SKFAC(net, loss_fn, optimizer, loss_scale_manager, metrics)

    # Train loop
    for epoch in range(config.epoch_size):
        model.train()  # Set the model to training mode
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(inputs)

            # Compute loss
            loss = loss_fn(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update metrics if necessary

        # Evaluate metrics, perform any necessary logging or callbacks, etc.
