import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.crossentropy import CrossEntropy
from src.config import config_gpu as config
from src.dataset import create_dataset
from src.resnet_skfac import resnet50 as resnet

parser = argparse.ArgumentParser(description='Image classification')
parser.add_argument('--checkpoint_path', type=str, default='checkpoint_skfac/resnet-{}_{}.ckpt', help='Checkpoint file path')
parser.add_argument('--dataset_path', type=str, default='ImageNet2012/val', help='Dataset path')
args_opt = parser.parse_args()

if __name__ == '__main__':
    target = 'GPU'

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and target == 'GPU' else 'cpu')

    # Create dataset
    val_dataset = create_dataset(dataset_path=args_opt.dataset_path, do_train=False, batch_size=config.batch_size, target=target)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    # Define net
    net = resnet(class_num=config.class_num)
    net.eval()

    # Load checkpoint
    checkpoint = torch.load(args_opt.checkpoint_path)
    net.load_state_dict(checkpoint['state_dict'])

    # Define loss function
    loss_fn = CrossEntropy(smooth_factor=config.label_smooth_factor, num_classes=config.class_num)

    # Define metrics
    top_1_accuracy = nn.Accuracy()
    top_5_accuracy = nn.TopkAccuracy(k=5)

    # Evaluation loop
    net.to(device)
    net.eval()
    top_1_acc_list = []
    top_5_acc_list = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = net(inputs)

            # Compute loss
            loss = loss_fn(outputs, labels)

            # Compute top-1 and top-5 accuracies
            top_1_accuracy(outputs, labels)
            top_5_accuracy(outputs, labels)

        top_1_acc = top_1_accuracy.compute().item()
        top_5_acc = top_5_accuracy.compute().item()

        print(f"Top-1 Accuracy: {top_1_acc}, Top-5 Accuracy: {top_5_acc} ckpt= {args_opt.checkpoint_path}")
