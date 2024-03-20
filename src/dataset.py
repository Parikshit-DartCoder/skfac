import os
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

def create_dataset(dataset_path, do_train, repeat_num=1, batch_size=32):
    """
    Create a train or eval ImageNet2012 dataset for ResNet50.

    Args:
        dataset_path (string): The path of the dataset.
        do_train (bool): Whether dataset is used for train or eval.
        repeat_num (int): The repeat times of dataset. Default: 1.
        batch_size (int): The batch size of dataset. Default: 32.

    Returns:
        dataset (torch.utils.data.Dataset)
    """

    if do_train:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    dataset = ImageFolder(root=dataset_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

    return dataloader
